from __future__ import print_function
from collections import defaultdict, deque, Counter
import datetime
import time
import torch
import torch.distributed as dist
import errno
import os
import sys
import argparse
import traceback
import numpy as np
from typing import Sequence, Tuple, Dict, Any, List


# ---------------------------
# Device helpers (NPU/CUDA/CPU)
# ---------------------------
def _is_npu_available():
    return hasattr(torch, "npu") and torch.npu.is_available()

def _is_cuda_available():
    return torch.cuda.is_available()

def _device_str():
    if _is_npu_available():
        return "npu"
    if _is_cuda_available():
        return "cuda"
    return "cpu"

def _max_memory_allocated_mb():
    MB = 1024.0 * 1024.0
    try:
        if _is_npu_available():
            return torch.npu.max_memory_allocated() / MB
        if _is_cuda_available():
            return torch.cuda.max_memory_allocated() / MB
    except Exception:
        pass
    return 0.0

def _synchronize():
    try:
        if _is_npu_available():
            torch.npu.synchronize()
        elif _is_cuda_available():
            torch.cuda.synchronize()
    except Exception:
        pass


# ---------------------------
# Smoothed metrics
# ---------------------------
class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a window or the global avg."""
    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        if not is_dist_avail_and_initialized():
            return
        dev = _device_str()
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device=dev)
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / max(1, self.count)

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = self.delimiter.join([
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}',
            'max mem: {memory:.0f}'
        ])
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                print(log_msg.format(
                    i, len(iterable), eta=eta_string,
                    meters=str(self),
                    time=str(iter_time), data=str(data_time),
                    memory=_max_memory_allocated_mb()
                ))
                sys.stdout.flush()
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str}')


# ---------------------------
# Extra evaluation utils
# ---------------------------
class MetricMeter:
    """Track IoU / Precision / Recall / F1 for segmentation"""
    def __init__(self, iou_thresholds: Sequence[float]):
        self.thr = list(iou_thresholds)
        self.correct_at = np.zeros(len(self.thr), dtype=np.int64)
        self.n_samples = 0
        self.per_sample_iou = []
        self.cum_I_fg = 0
        self.cum_U_fg = 0
        self.I_per_class = np.zeros(2, dtype=np.int64)
        self.U_per_class = np.zeros(2, dtype=np.int64)
        self.TP = 0
        self.FP = 0
        self.FN = 0

    def update(self, pred: torch.Tensor, gt: torch.Tensor):
        B = pred.size(0)
        self.n_samples += B
        pred_fg = (pred == 1)
        gt_fg = (gt == 1)
        inter_fg = (pred_fg & gt_fg).sum(dim=(1, 2)).float()
        union_fg = (pred_fg | gt_fg).sum(dim=(1, 2)).float()
        iou_fg_sample = torch.where(union_fg > 0, inter_fg / union_fg, torch.zeros_like(union_fg))
        self.per_sample_iou.extend(iou_fg_sample.detach().cpu().tolist())
        for i, thr in enumerate(self.thr):
            self.correct_at[i] += int((iou_fg_sample >= thr).sum().item())
        self.cum_I_fg += int(inter_fg.sum().item())
        self.cum_U_fg += int(union_fg.sum().item())
        pred_bg = ~pred_fg
        gt_bg = ~gt_fg
        inter_bg = (pred_bg & gt_bg).sum(dim=(1, 2)).long()
        union_bg = (pred_bg | gt_bg).sum(dim=(1, 2)).long()
        self.I_per_class[1] += int(inter_fg.sum().item())
        self.U_per_class[1] += int(union_fg.sum().item())
        self.I_per_class[0] += int(inter_bg.sum().item())
        self.U_per_class[0] += int(union_bg.sum().item())
        self.TP += int((pred_fg & gt_fg).sum().item())
        self.FP += int((pred_fg & ~gt_fg).sum().item())
        self.FN += int((~pred_fg & gt_fg).sum().item())

    def summarize(self) -> Dict[str, float]:
        mIoU = float(np.mean(self.per_sample_iou)) if self.per_sample_iou else 0.0
        oIoU = (self.cum_I_fg / self.cum_U_fg) if self.cum_U_fg > 0 else 0.0
        class_ious = []
        for c in (0, 1):
            if self.U_per_class[c] > 0:
                class_ious.append(self.I_per_class[c] / self.U_per_class[c])
        cIoU = float(np.mean(class_ious)) if class_ious else 0.0
        precision_fg = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0.0
        recall_fg = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0.0
        f1_fg = (2 * precision_fg * recall_fg / (precision_fg + recall_fg)
                 if (precision_fg + recall_fg) > 0 else 0.0)
        precision = {f"precision@{t:.1f}": (self.correct_at[i] * 100.0 / max(self.n_samples, 1))
                     for i, t in enumerate(self.thr)}
        out = {
            "mIoU": mIoU * 100.0,
            "oIoU": oIoU * 100.0,
            "cIoU": cIoU * 100.0,
            "Precision": precision_fg * 100.0,
            "Recall": recall_fg * 100.0,
            "F1": f1_fg * 100.0,
        }
        out.update(precision)
        return out


def _to_namespace(x: Any) -> argparse.Namespace:
    if isinstance(x, argparse.Namespace):
        ns = x
    elif isinstance(x, dict):
        ns = argparse.Namespace(**x)
    elif hasattr(x, "__dict__"):
        ns = argparse.Namespace(**vars(x))
    else:
        ns = argparse.Namespace()
    defaults = {
        "mha": "",
        "swin_type": "large",
        "window12": False,
        "fusion_drop": 0.0,
        "pretrained_swin_weights": "",
        "model": "lavt",
        "prefusion_model": "unet_fuser",
        "img_size": 768,
        "ck_bert": "bert-base-uncased",
        "bert_tokenizer": "bert-base-uncased",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }
    for k, v in defaults.items():
        if not hasattr(ns, k):
            setattr(ns, k, v)
    return ns


def _print_ckpt_overview(ckpt: Dict[str, Any], max_list: int = 30):
    keys = list(ckpt.keys())
    print(f"[ckpt] top-level keys: {keys}")
    if "args" in ckpt:
        try:
            args = _to_namespace(ckpt["args"])
            print("[ckpt][args] =====")
            for k in sorted(vars(args).keys()):
                print(f"  - {k}: {getattr(args, k)}")
            print("===================")
        except Exception:
            print("[ckpt] WARNING: failed to pretty-print args")
            traceback.print_exc()
    if "model" in ckpt and isinstance(ckpt["model"], dict):
        state = ckpt["model"]
        print(f"[ckpt][model] num params: {len(state)}")
        prefix_counter = Counter(k.split('.')[0] for k in state.keys())
        print(f"[ckpt][model] top-level prefix counts: {dict(prefix_counter)}")
        interest = [k for k in state.keys() if (
            "relative_position_bias_table" in k or
            "relative_position_index" in k or
            "patch_embed.proj" in k or
            "layers.0.blocks.0.attn" in k
        )]
        print(f"[ckpt][model] show {min(max_list, len(interest))} interesting keys (shape):")
        for k in interest[:max_list]:
            try:
                print(f"  - {k}: {tuple(state[k].shape)}")
            except Exception:
                print(f"  - {k}: <shape unavailable>")
    else:
        print("[ckpt] WARNING: 'model' not in checkpoint or not a dict.")


def _report_coverage(model: torch.nn.Module, state: Dict[str, torch.Tensor]):
    model_keys = set(model.state_dict().keys())
    ckpt_keys = set(state.keys())
    miss = sorted(list(model_keys - ckpt_keys))
    extra = sorted(list(ckpt_keys - model_keys))
    covered = len(model_keys) - len(miss)
    print(f"[coverage] loaded keys: {covered} / model params: {len(model_keys)}  (coverage={covered/len(model_keys):.3f})")
    if miss:
        print(f"[coverage] missing (kept init) = {len(miss)} (show up to 20): {miss[:20]}")
    if extra:
        print(f"[coverage] unexpected (ignored) = {len(extra)} (show up to 20): {extra[:20]}")


# ---------------------------
# Distributed helpers
# ---------------------------
def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise

def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print
    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)
    __builtin__.print = print

def is_dist_avail_and_initialized():
    # Use torch.distributed primitives correctly
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()

def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()

def is_main_process():
    return get_rank() == 0

def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)

def init_distributed_mode(args):
    rank_env = os.environ.get("RANK", None)
    world_env = os.environ.get("WORLD_SIZE", None)
    local_rank_env = os.environ.get("LOCAL_RANK", None)
    local_rank = getattr(args, "local_rank", None)
    if local_rank is None and local_rank_env is not None:
        try:
            local_rank = int(local_rank_env)
            args.local_rank = local_rank
        except Exception:
            local_rank = None

    need_dist = (rank_env is not None) and (world_env is not None)
    use_npu = _is_npu_available()
    use_cuda = _is_cuda_available()

    if not need_dist:
        args.distributed = False
        args.rank = 0
        args.world_size = 1
        if use_npu:
            dev = torch.device(f"npu:{local_rank if local_rank is not None else 0}")
            torch.npu.set_device(dev)
        elif use_cuda:
            dev = torch.device(f"cuda:{local_rank if local_rank is not None else 0}")
            torch.cuda.set_device(dev)
        else:
            dev = torch.device("cpu")
        args.device = dev
        if args.output_dir:
            mkdir(args.output_dir)
        if getattr(args, "model_id", None):
            mkdir(os.path.join('./models/', args.model_id))
        setup_for_distributed(is_main_process())
        return

    args.rank = int(rank_env)
    args.world_size = int(world_env)
    args.distributed = True

    if use_npu:
        dev = torch.device(f"npu:{local_rank}")
        torch.npu.set_device(dev)
        backend = "hccl"
    elif use_cuda:
        dev = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(dev)
        backend = "nccl"
    else:
        dev = torch.device("cpu")
        backend = "gloo"

    args.device = dev
    dist.init_process_group(
        backend=backend, init_method="env://",
        world_size=args.world_size, rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(is_main_process())
    if args.output_dir:
        mkdir(args.output_dir)
    if getattr(args, "model_id", None):
        mkdir(os.path.join('./models/', args.model_id))

if not hasattr(dist, "is_dist_avail_and_initialized"):
    def _compat_is_dist_avail_and_initialized():
        return dist.is_available() and dist.is_initialized()
    dist.is_dist_avail_and_initialized = _compat_is_dist_avail_and_initialized