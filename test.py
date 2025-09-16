# test.py
import os
import io
import argparse
from typing import Tuple, Sequence, Dict, Any
import traceback
import numpy as np
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn.functional as F
import torch.utils.data as data

# ---- Project modules ----
from bert.modeling_bert import BertModel
from lib import segmentation
from prefusion import build_prefusion
from data.refer_dataset import FusionParquetDataset, PairedTwoImageTransform

# ---- Utils (moved out from here) ----
from utils import MetricMeter, _to_namespace, _print_ckpt_overview, _report_coverage


# -----------------------
# Visualization helpers
# -----------------------
def overlay_red(pil_img: Image.Image, mask01: np.ndarray, alpha: int = 110) -> Image.Image:
    """Overlay red transparent mask (mask=1) on pil_img"""
    if mask01.dtype != np.uint8:
        mask01 = mask01.astype(np.uint8)
    H, W = pil_img.size[1], pil_img.size[0]
    if mask01.shape != (H, W):
        mask_img = Image.fromarray((mask01 * 255).astype(np.uint8)).resize((W, H), resample=Image.NEAREST)
        mask01 = (np.array(mask_img) > 127).astype(np.uint8)
    base = pil_img.convert("RGBA")
    a = (mask01 * alpha).astype(np.uint8)
    overlay = np.dstack([np.full_like(a, 255), np.zeros_like(a), np.zeros_like(a), a])  # red channel
    overlay = Image.fromarray(overlay, mode="RGBA")
    return Image.alpha_composite(base, overlay).convert("RGB")


def denorm_vis(std_img: torch.Tensor, prefusion_model: torch.nn.Module) -> torch.Tensor:
    """De-normalize image using prefusion mean/std"""
    mean = getattr(prefusion_model, "mean_vis").to(std_img.device)
    std = getattr(prefusion_model, "std_vis").to(std_img.device)
    return torch.clamp(std_img * std + mean, 0.0, 1.0)


def _pil_from_bytes(x: Any) -> Image.Image:
    if isinstance(x, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(x))).convert("RGB")
    raise TypeError(f"image must be bytes-like, got {type(x)}")


def _mask_from_npy_bytes(x: Any) -> np.ndarray:
    if not isinstance(x, (bytes, bytearray, memoryview)):
        raise TypeError(f"segmentation must be NPY bytes, got {type(x)}")
    arr = np.load(io.BytesIO(bytes(x)), allow_pickle=False)
    return (arr > 0).astype(np.uint8)


# -----------------------
# Save visualizations
# -----------------------
def save_visuals(
    fused01: torch.Tensor,
    pred: torch.Tensor,
    gt: torch.Tensor,
    df,
    step: int,
    global_base_idx: int,
    out_fusion: str,
    out_bin: str,
    out_overlay: str,
    out_gt: str,
    Hf: int,
    Wf: int,
):
    """Save fusion, binary, overlay, and GT results for each sample in a batch"""
    B = fused01.size(0)
    for b in range(B):
        if df is not None:
            row = df.iloc[global_base_idx + b]
            name = str(row.get("image_name", f"{step:06d}_{b}"))
            vis_orig = _pil_from_bytes(row["visible_image"])
            orig_W, orig_H = vis_orig.size
            try:
                idx = int(row.get("index"))
            except Exception:
                idx = int(global_base_idx + b)
            try:
                gt_orig01 = _mask_from_npy_bytes(row["segmentation"])
            except Exception:
                gt_orig01 = None
        else:
            name = f"{step:06d}_{b}"
            orig_W, orig_H = Wf, Hf
            idx = int(global_base_idx + b)
            gt_orig01 = None

        # Fusion image
        arr01 = fused01[b].clamp(0, 1).mul(255).byte().permute(1, 2, 0).cpu().numpy()
        fused_pil = Image.fromarray(arr01)
        if (fused_pil.size[0], fused_pil.size[1]) != (orig_W, orig_H):
            fused_pil = fused_pil.resize((orig_W, orig_H), resample=Image.BILINEAR)
        fused_pil.save(os.path.join(out_fusion, f"{name}.jpg"), quality=95)

        # Binary prediction
        pred_np = pred[b].detach().cpu().numpy().astype(np.uint8) * 255
        pred_pil = Image.fromarray(pred_np)
        if pred_pil.size != (orig_W, orig_H):
            pred_pil = pred_pil.resize((orig_W, orig_H), resample=Image.NEAREST)
        pred_pil.save(os.path.join(out_bin, f"{idx}.png"))

        # Overlay
        pred01_resized = (np.array(pred_pil) > 127).astype(np.uint8)
        pred_vis = overlay_red(fused_pil, pred01_resized, alpha=110)
        pred_vis.save(os.path.join(out_overlay, f"{idx}.jpg"), quality=95)

        # GT overlay
        if gt_orig01 is None:
            gt_np = gt[b].detach().cpu().numpy().astype(np.uint8) * 255
            gt_pil = Image.fromarray(gt_np)
            if gt_pil.size != (orig_W, orig_H):
                gt_pil = gt_pil.resize((orig_W, orig_H), resample=Image.NEAREST)
            gt01_vis = (np.array(gt_pil) > 127).astype(np.uint8)
        else:
            gt01_vis = gt_orig01.astype(np.uint8)
        gt_vis_img = overlay_red(fused_pil, gt01_vis, alpha=110)
        gt_vis_img.save(os.path.join(out_gt, f"{idx}.jpg"), quality=95)


# -----------------------
# Main test pipeline
# -----------------------
@torch.no_grad()
def run_test(
    ckpt_path: str,
    test_parquet: str,
    out_root: str,
    model_name: str = "lavt",
    prefusion_name: str = "unet_fuser",
    img_size: int = 480,
    batch_size: int = 1,
    num_workers: int = 1,
    bert_tokenizer: str = "bert-base-uncased",
    ck_bert: str = "bert-base-uncased",
    pretrained_swin_weights: str | None = None,
    device: str = None,
    iou_thresholds: Tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
):
    # Device
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else ("npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu"))
    dev = torch.device(device)
    if device.startswith("npu"):
        try:
            import torch_npu  # noqa
            torch.npu.set_device(dev)
        except Exception:
            pass

    # Load checkpoint
    print(f"[Load] checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    _print_ckpt_overview(ckpt, max_list=40)
    if "args" not in ckpt:
        raise RuntimeError("Checkpoint missing 'args'. Cannot rebuild model.")
    train_args = _to_namespace(ckpt["args"])

    if getattr(train_args, "ck_bert", None) is None:
        train_args.ck_bert = ck_bert
    if getattr(train_args, "bert_tokenizer", None) is None:
        train_args.bert_tokenizer = bert_tokenizer

    # Segmentation model
    seg_name = getattr(train_args, "model", model_name)
    swin_pretrained = getattr(train_args, "pretrained_swin_weights", "")
    if not (isinstance(swin_pretrained, str) and os.path.isfile(swin_pretrained)):
        swin_pretrained = ""
    seg_model = segmentation.__dict__[seg_name](pretrained=swin_pretrained, args=train_args)

    if "model" not in ckpt:
        raise RuntimeError("Checkpoint missing 'model' weights.")
    try:
        seg_model.load_state_dict(ckpt["model"], strict=True)
        print("[seg] state_dict loaded with strict=True.")
    except Exception:
        print("[seg] Exception while loading state_dict (strict=True):")
        traceback.print_exc()
        _report_coverage(seg_model, ckpt["model"])
        raise
    _report_coverage(seg_model, ckpt["model"])
    seg_model.to(dev).eval()

    # Prefusion
    pf_name = getattr(train_args, "prefusion_model", prefusion_name)
    prefusion_model = build_prefusion(pf_name, args=train_args).to(dev).eval()
    if "prefusion_model" in ckpt:
        try:
            prefusion_model.load_state_dict(ckpt["prefusion_model"], strict=False)
            print("[prefusion] state_dict loaded with strict=False.")
        except Exception:
            print("[prefusion] Exception while loading state_dict:")
            traceback.print_exc()
            _report_coverage(prefusion_model, ckpt["prefusion_model"])

    # BERT
    bert_model = BertModel.from_pretrained(getattr(train_args, "ck_bert", ck_bert))
    bert_model.pooler = None
    if "bert_model" in ckpt:
        try:
            bert_model.load_state_dict(ckpt["bert_model"], strict=False)
            print("[bert] state_dict loaded with strict=False.")
        except Exception:
            print("[bert] Exception while loading state_dict:")
            traceback.print_exc()
            _report_coverage(bert_model, ckpt["bert_model"])
    bert_model.to(dev).eval()

    # Dataset
    tf_img_size = int(getattr(train_args, "img_size", img_size))
    tfm = PairedTwoImageTransform(size=(tf_img_size, tf_img_size))
    dataset = FusionParquetDataset(
        parquet_path_or_dir=test_parquet,
        bert_tokenizer=getattr(train_args, "bert_tokenizer", bert_tokenizer),
        image_transforms=tfm,
        max_tokens=20,
        eval_mode=True,
    )
    loader = data.DataLoader(dataset, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=True)
    df = getattr(dataset, "df", None)

    # Output dirs
    out_fusion = os.path.join(out_root, "fusion")
    out_gt = os.path.join(out_root, "GT")
    out_bin = os.path.join(out_root, "Binary")
    out_overlay = os.path.join(out_root, "Overlay")
    for d in [out_fusion, out_gt, out_bin, out_overlay]:
        os.makedirs(d, exist_ok=True)

    # Metrics
    meter = MetricMeter(iou_thresholds)

    # Loop
    global_base_idx = 0
    for step, batch in enumerate(tqdm(loader, desc="Testing", leave=True)):
        vis_img, ir_img, target, sentences, attentions = batch
        vis_img = vis_img.to(dev, non_blocking=True)
        ir_img = ir_img.to(dev, non_blocking=True)
        target = target.to(dev, non_blocking=True)
        sentences = sentences.to(dev, non_blocking=True).squeeze(1)  # (B, T)
        attentions = attentions.to(dev, non_blocking=True).squeeze(1)  # (B, T)

        # ---------- DEBUG: print once for the first batch ----------
        if step == 0:
            print(f"[DEBUG][step=0] vis_img: {tuple(vis_img.shape)} {vis_img.dtype} on {vis_img.device}")
            print(f"[DEBUG][step=0] ir_img : {tuple(ir_img.shape)} {ir_img.dtype} on {ir_img.device}")
            print(f"[DEBUG][step=0] sentences: {tuple(sentences.shape)} attentions: {tuple(attentions.shape)}")
            print(f"[DEBUG][step=0] attention nonzero per-sample: {attentions.sum(dim=1).tolist()}")

        # ---- Text encoding FIRST (so that pre-fusion can use it) ----
        last_hidden = bert_model(sentences, attention_mask=attentions)[0]  # (B, T, 768)
        # For LAVT seg model we also build embedding and mask later
        embedding = last_hidden.permute(0, 2, 1)  # (B, 768, T)
        l_mask = attentions.unsqueeze(-1)         # (B, T, 1)

        # ---------- Prefusion (now WITH text) ----------
        # Pass text_feats (B, T, 768) and text_mask (B, T) into the fuser
        fused = prefusion_model(vis_img, ir_img, text_feats=last_hidden, text_mask=attentions)

        # Extra runtime checks if text unexpectedly missing
        if step == 0 or (last_hidden is None):
            print(f"[DEBUG] Using text for pre-fusion: {'YES' if last_hidden is not None else 'NO'}")
            if last_hidden is not None:
                print(f"[DEBUG] text_feats: {tuple(last_hidden.shape)}; text_mask sum: {int(attentions.sum().item())}")

        B, _, Hf, Wf = fused.shape

        # ---- Segmentation ----
        logits = seg_model(fused, embedding, l_mask=l_mask)
        if logits.shape[-2:] != (Hf, Wf):
            logits = F.interpolate(logits, size=(Hf, Wf), mode="bilinear", align_corners=False)

        if logits.size(1) == 2:
            pred = logits.argmax(1)
        else:
            pred = (logits[:, 0] > 0).to(torch.uint8)
        gt = target.long()

        # Visualization
        fused01 = denorm_vis(fused, prefusion_model)
        save_visuals(fused01, pred, gt, df, step, global_base_idx,
                     out_fusion, out_bin, out_overlay, out_gt, Hf, Wf)

        # Metrics
        meter.update(pred=pred, gt=gt)
        global_base_idx += B

    # Results
    results = meter.summarize()
    print("\n==== Test Results ====")
    for k, v in results.items():
        if k.startswith("precision@"):
            print(f"{k}: {v:.2f}")
    print(f"mIoU: {results['mIoU']:.2f}")
    print(f"oIoU: {results['oIoU']:.2f}")
    print(f"cIoU: {results['cIoU']:.2f}")
    print(f"Precision: {results['Precision']:.2f}")
    print(f"Recall: {results['Recall']:.2f}")
    print(f"F1: {results['F1']:.2f}")
    return results


# -----------------------
# CLI
# -----------------------
def build_argparser():
    p = argparse.ArgumentParser("RIS-Fusion Test (resume-style)")
    p.add_argument("--ckpt", required=True, help="checkpoint path (must contain 'model' / 'prefusion_model' / 'args')")
    p.add_argument("--test_parquet", required=True, help="test parquet path or dir")
    p.add_argument("--out_dir", default="test_results", help="output dir root")
    p.add_argument("--model", default="lavt")
    p.add_argument("--prefusion_model", default="unet_fuser")
    p.add_argument("--img_size", type=int, default=480)
    p.add_argument("-b", "--batch_size", type=int, default=4)
    p.add_argument("-j", "--workers", type=int, default=4)
    p.add_argument("--bert_tokenizer", default="bert-base-uncased")
    p.add_argument("--ck_bert", default="bert-base-uncased")
    p.add_argument("--pretrained_swin_weights", default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--iou_thresholds", default="0.5,0.6,0.7,0.8,0.9")
    return p


def parse_thresholds(s: str) -> Tuple[float, ...]:
    return tuple(float(x) for x in s.split(",") if x.strip())


if __name__ == "__main__":
    parser = build_argparser()
    args = parser.parse_args()
    dev = (args.device or
           ("cuda" if torch.cuda.is_available()
            else ("npu" if (hasattr(torch, "npu") and torch.npu.is_available()) else "cpu")))
    run_test(
        ckpt_path=args.ckpt,
        test_parquet=args.test_parquet,
        out_root=args.out_dir,
        model_name=args.model,
        prefusion_name=args.prefusion_model,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
        bert_tokenizer=args.bert_tokenizer,
        ck_bert=args.ck_bert,
        pretrained_swin_weights=args.pretrained_swin_weights,
        device=dev,
        iou_thresholds=parse_thresholds(args.iou_thresholds),
    )
