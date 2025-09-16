# train.py

import os, time, datetime, gc
from functools import reduce
import operator

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data

import numpy as np
from PIL import Image

from bert.modeling_bert import BertModel
from lib import segmentation
import transforms as T
import utils

from data.refer_dataset import FusionParquetDataset, PairedTwoImageTransform
from prefusion import build_prefusion
from losses.fusion_losses import build_loss


# -----------------------
# Dataset
# -----------------------
def get_dataset(image_set, transform, args):
    parquet_path = args.train_parquet if image_set == "train" else args.val_parquet
    if parquet_path is None:
        raise ValueError(f"Please set --{'train_parquet' if image_set=='train' else 'val_parquet'}")

    paired_tf = PairedTwoImageTransform(size=(args.img_size, args.img_size))
    ds = FusionParquetDataset(
        parquet_path_or_dir=parquet_path,
        bert_tokenizer=args.ck_bert,
        image_transforms=paired_tf,
        max_tokens=20,
        eval_mode=(image_set != "train"),
    )
    return ds, 2  # num_classes=2


def get_transform(args):
    return T.Compose([
        T.Resize(args.img_size, args.img_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225])
    ])


# -----------------------
# Metrics
# -----------------------
def IoU(logits, gt):
    pred = logits.argmax(1)
    intersection = torch.sum(pred * gt)
    union = torch.sum(pred + gt) - intersection
    if intersection == 0 or union == 0:
        return 0.0, intersection, union
    return float(intersection) / float(union), intersection, union


# -----------------------
# Losses
# -----------------------
def bce_seg_loss_area_norm(seg_logits, target, eps: float = 1e-6):
    B, C, h, w = seg_logits.shape
    H, W = target.shape[-2:]
    if (h, w) != (H, W):
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)

    fg_logit = seg_logits[:, 1:2] if C == 2 else seg_logits
    tgt = target.float().unsqueeze(1)

    per_pix = F.binary_cross_entropy_with_logits(fg_logit, tgt, reduction='none')
    pos_mask, neg_mask = tgt, 1.0 - tgt

    pos_sum, neg_sum = (per_pix * pos_mask).sum((1,2,3)), (per_pix * neg_mask).sum((1,2,3))
    pos_cnt, neg_cnt = pos_mask.sum((1,2,3)), neg_mask.sum((1,2,3))

    pos_mean = pos_sum / pos_cnt.clamp_min(1.0)
    neg_mean = neg_sum / neg_cnt.clamp_min(1.0)

    use_pos, use_neg = (pos_cnt > eps).float(), (neg_cnt > eps).float()
    denom = (use_pos + use_neg).clamp_min(1.0)
    loss = (use_pos * pos_mean + use_neg * neg_mean) / denom
    return loss.mean()


def dice_loss(seg_logits, target, eps: float = 1e-6):
    B, C, h, w = seg_logits.shape
    H, W = target.shape[-2:]
    if (h, w) != (H, W):
        seg_logits = F.interpolate(seg_logits, size=(H, W), mode="bilinear", align_corners=False)

    fg_logit = seg_logits[:, 1:2] if C == 2 else seg_logits
    p_fg = torch.sigmoid(fg_logit)
    tgt = target.float().unsqueeze(1)

    inter_pos = (p_fg * tgt).sum((1,2,3))
    sum_pos = p_fg.sum((1,2,3)) + tgt.sum((1,2,3))
    dice_pos = (2 * inter_pos + eps) / (sum_pos + eps)

    p_bg, tgt_bg = 1.0 - p_fg, 1.0 - tgt
    inter_bg = (p_bg * tgt_bg).sum((1,2,3))
    sum_bg = p_bg.sum((1,2,3)) + tgt_bg.sum((1,2,3))
    dice_bg = (2 * inter_bg + eps) / (sum_bg + eps)

    pos_cnt, bg_cnt = tgt.sum((1,2,3)), tgt_bg.sum((1,2,3))
    use_pos, use_bg = (pos_cnt > 0).float(), (bg_cnt > 0).float()
    denom = (use_pos + use_bg).clamp_min(1.0)

    dice = (use_pos * dice_pos + use_bg * dice_bg) / denom
    return (1.0 - dice).mean()


# -----------------------
# Text utils
# -----------------------
def _prepare_text_2d(sentences, attentions):
    B = sentences.size(0)
    if sentences.dim() == 3 and sentences.size(1) == 1:
        sentences = sentences.squeeze(1)
    elif sentences.dim() > 2:
        sentences = sentences.view(B, -1)
    sentences = sentences.to(torch.long)

    if attentions.dim() == 3 and attentions.size(1) == 1:
        attentions = attentions.squeeze(1)
    elif attentions.dim() > 2:
        attentions = attentions.view(B, -1)
    attentions = (attentions > 0).to(torch.long)

    assert sentences.dim() == 2 and attentions.dim() == 2
    return sentences, attentions


# -----------------------
# Evaluation
# -----------------------
def evaluate(model, data_loader, bert_model, prefusion_model, device,
             save_dir: str | None = None, save_limit: int | None = None):
    model.eval()
    prefusion_model.eval()

    def _denorm_vis(x): return torch.clamp(x * prefusion_model.std_vis + prefusion_model.mean_vis, 0, 1)

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    acc_ious, total_its, cum_I, cum_U = 0, 0, 0, 0
    eval_seg_iou_list = [.5, .6, .7, .8, .9]
    seg_correct, seg_total = np.zeros(len(eval_seg_iou_list)), 0
    mean_IoU = []

    with torch.no_grad():
        for step, data in enumerate(metric_logger.log_every(data_loader, 100, header)):
            total_its += 1
            vis_img, ir_img, target, sentences, attentions = data
            vis_img, ir_img, target = vis_img.to(device), ir_img.to(device), target.to(device)
            sentences, attentions = sentences.to(device), attentions.to(device)

            fused = prefusion_model(vis_img, ir_img)
            sentences, attentions = sentences.squeeze(1), attentions.squeeze(1)

            if bert_model is not None:
                last_hidden = bert_model(sentences, attention_mask=attentions)[0]
                embedding = last_hidden.permute(0, 2, 1)
                attentions = attentions.unsqueeze(-1)
                logits = model(fused, embedding, l_mask=attentions)
            else:
                logits = model(fused, sentences, l_mask=attentions)

            iou, I, U = IoU(logits, target)
            acc_ious += iou
            mean_IoU.append(iou)
            cum_I, cum_U = cum_I + I, cum_U + U
            for i, thr in enumerate(eval_seg_iou_list):
                seg_correct[i] += (iou >= thr)
            seg_total += 1

    mIoU = float(np.mean(mean_IoU)) if mean_IoU else 0.0
    print(f'Final results:\nMean IoU: {mIoU*100:.2f}')
    for thr, correct in zip(eval_seg_iou_list, seg_correct):
        print(f'precision@{thr}: {correct * 100 / max(seg_total, 1):.2f}')
    print(f'Overall IoU: {cum_I * 100 / max(cum_U,1):.2f}')

    return 100 * acc_ious / max(total_its, 1), 100 * cum_I / max(cum_U, 1)


# -----------------------
# Train loop
# -----------------------
def train_one_epoch(model, bert_model, prefusion_model, data_loader,
                    opt_seg, opt_pf, sch_seg, sch_pf, epoch, device,
                    print_freq, fusion_loss_fn, mask_weight_cfg=None):
    model.train()
    prefusion_model.train()
    if bert_model is not None:
        bert_model.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr_seg', utils.SmoothedValue(window_size=1, fmt='{value}'))
    metric_logger.add_meter('lr_pf',  utils.SmoothedValue(window_size=1, fmt='{value}'))
    header = f'Epoch: [{epoch}]'

    for data in metric_logger.log_every(data_loader, print_freq, header):
        vis_img, ir_img, target, sentences, attentions = data
        vis_img, ir_img, target = vis_img.to(device), ir_img.to(device), target.to(device)
        sentences, attentions = sentences.to(device), attentions.to(device)

        if bert_model is not None:
            sentences, attentions = _prepare_text_2d(sentences, attentions)
            last_hidden = bert_model(sentences, attention_mask=attentions)[0]
            embedding = last_hidden.permute(0, 2, 1)
        else:
            embedding = sentences

        fused = prefusion_model(vis_img, ir_img,
                                text_feats=last_hidden if bert_model else None,
                                text_mask=attentions if bert_model else None)

        seg_logits = model(fused, embedding, l_mask=attentions.unsqueeze(-1) if bert_model else attentions)

        seg_loss = dice_loss(seg_logits, target)
        pf_raw, pf_stats = fusion_loss_fn.prefusion_loss(fused, vis_img, ir_img)
        pf_loss = fusion_loss_fn.lambda_pf * pf_raw

        opt_seg.zero_grad(set_to_none=True)
        opt_pf.zero_grad(set_to_none=True)
        (seg_loss + pf_loss).backward()
        opt_seg.step()
        opt_pf.step()
        sch_seg.step()
        sch_pf.step()

        metric_logger.update(
            loss=(seg_loss + pf_loss).item(),
            seg=seg_loss.item(),
            pf=pf_loss.item(),
            lr_seg=opt_seg.param_groups[0]["lr"],
            lr_pf=opt_pf.param_groups[0]["lr"],
            **pf_stats
        )

        del last_hidden, embedding, fused, seg_logits
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        if hasattr(torch, "npu") and torch.npu.is_available(): torch.npu.empty_cache()


# -----------------------
# Main
# -----------------------
def main(args):
    utils.init_distributed_mode(args)
    device = args.device
    print(f"[Device] {device} | distributed={getattr(args,'distributed',False)} "
          f"| rank={getattr(args,'rank',-1)} | world_size={getattr(args,'world_size',1)}")

    dataset, _ = get_dataset("train", get_transform(args), args)
    dataset_test, _ = get_dataset("val", get_transform(args), args)

    if getattr(args, "distributed", False):
        num_tasks = utils.get_world_size()
        rank = utils.get_rank()
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, num_replicas=num_tasks, rank=rank)
    else:
        train_sampler = torch.utils.data.RandomSampler(dataset)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.workers, pin_memory=args.pin_mem, drop_last=True
    )
    data_loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=1, sampler=test_sampler,
                                                   num_workers=args.workers)

    seg_model = segmentation.__dict__[args.model](pretrained=args.pretrained_swin_weights, args=args)
    if getattr(args, "distributed", False):
        seg_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(seg_model)
    seg_model.to(device)
    single_model = seg_model.module if getattr(args, "distributed", False) else seg_model

    prefusion_model = build_prefusion(args.prefusion_model, args=args).to(device)
    if getattr(args, "distributed", False):
        prefusion_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(prefundion_model)  # type: ignore
    single_prefusion = prefusion_model.module if getattr(args,"distributed",False) else prefusion_model

    if args.model != 'lavt_one':
        bert_model = BertModel.from_pretrained(args.ck_bert)
        bert_model.pooler = None
        if getattr(args,"distributed",False):
            bert_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(bert_model)
        bert_model.to(device)
        single_bert = bert_model.module if getattr(args,"distributed",False) else bert_model
    else:
        bert_model, single_bert = None, None

    seg_params = [{"params": [p for p in single_model.parameters() if p.requires_grad]}]
    if single_bert is not None:
        seg_params.append({"params": [p for p in single_bert.parameters() if p.requires_grad]})
    pf_params = [p for p in single_prefusion.parameters() if p.requires_grad]

    lr_seg = getattr(args, "lr_seg", args.lr)
    lr_pf = getattr(args, "lr_pf", 1e-4)
    wd_seg = getattr(args, "wd_seg", args.weight_decay)
    wd_pf = getattr(args, "wd_pf", 0.0)

    opt_seg = torch.optim.AdamW(seg_params, lr=lr_seg, weight_decay=wd_seg, amsgrad=args.amsgrad)
    opt_pf = torch.optim.AdamW(pf_params, lr=lr_pf, weight_decay=wd_pf, amsgrad=args.amsgrad)

    total_steps = len(data_loader) * max(args.epochs, 1)
    lr_lambda = lambda x: (1 - x / max(total_steps, 1)) ** 0.9
    sch_seg, sch_pf = torch.optim.lr_scheduler.LambdaLR(opt_seg, lr_lambda), torch.optim.lr_scheduler.LambdaLR(opt_pf, lr_lambda)

    fusion_loss_fn = build_loss(args, device, single_prefusion).to(device)

    best_oIoU, start_time = -0.1, time.time()
    for epoch in range(args.epochs):
        if getattr(args,"distributed",False) and hasattr(data_loader.sampler, "set_epoch"):
            data_loader.sampler.set_epoch(epoch)

        train_one_epoch(seg_model, bert_model, prefusion_model, data_loader,
                        opt_seg, opt_pf, sch_seg, sch_pf,
                        epoch, device, args.print_freq,
                        fusion_loss_fn)

        iou, oIoU = evaluate(seg_model, data_loader_test, bert_model, prefusion_model, device=device)

        if iou > 30.0:
            to_save = {'model': single_model.state_dict(),
                       'prefusion_model': single_prefusion.state_dict(),
                       'epoch': epoch, 'args': args}
            if single_bert: to_save['bert_model'] = single_bert.state_dict()
            utils.save_on_master(to_save, os.path.join(args.output_dir, f'{epoch}.pth'))

        if oIoU > best_oIoU:
            utils.save_on_master({'model': single_model.state_dict(),
                                  'prefusion_model': single_prefusion.state_dict(),
                                  'epoch': epoch, 'args': args},
                                 os.path.join(args.output_dir, f'model_best_{args.model_id}.pth'))
            best_oIoU = oIoU

    print('Training time', str(datetime.timedelta(seconds=int(time.time()-start_time))))


if __name__ == "__main__":
    from args import get_parser
    parser = get_parser()
    args = parser.parse_args()
    main(args)
