# refer_dataset.py
import os, io, glob
from typing import Any, List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

from bert.tokenization_bert import BertTokenizer


# --------- Utility: load parquet(s) ----------
def _load_parquet_any(path_or_dir: str) -> pd.DataFrame:
    """
    Load a parquet file or all parquet files from a directory into a DataFrame.
    Ensures required columns are present and drops rows with NaN.
    """
    if os.path.isdir(path_or_dir):
        files = sorted(glob.glob(os.path.join(path_or_dir, "*.parquet")))
        if not files:
            raise FileNotFoundError(f"No parquet files under: {path_or_dir}")
        df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    else:
        df = pd.read_parquet(path_or_dir)

    required = ["image_name", "visible_image", "infrared_image", "question", "segmentation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in parquet: {missing}")

    df = df.dropna(subset=required).reset_index(drop=True)
    return df


# --------- Decode utilities: bytes -> PIL / ndarray ----------
def _pil_from_bytes(x: Any) -> Image.Image:
    """Decode RGB image from bytes-like object."""
    if isinstance(x, (bytes, bytearray, memoryview)):
        return Image.open(io.BytesIO(bytes(x))).convert("RGB")
    raise TypeError(f"image must be bytes-like, got {type(x)}")

def _mask_from_npy_bytes(x: Any) -> np.ndarray:
    """Decode binary segmentation mask (NPY bytes -> uint8 ndarray in {0,1})."""
    if not isinstance(x, (bytes, bytearray, memoryview)):
        raise TypeError(f"segmentation must be NPY bytes, got {type(x)}")
    arr = np.load(io.BytesIO(bytes(x)), allow_pickle=False)
    return (arr > 0).astype(np.uint8)


# --------- Default transform: pair of images + mask ----------
class PairedTwoImageTransform:
    """
    Apply the same geometric transforms to (vis_img, ir_img, mask):
      - Resize
      - ToTensor
      - Normalize
    Ensures spatial consistency across all three inputs.
    """
    def __init__(self,
                 size: Tuple[int, int] = (512, 512),
                 mean_vis=(0.485, 0.456, 0.406),
                 std_vis=(0.229, 0.224, 0.225),
                 mean_ir=(0.485, 0.456, 0.406),
                 std_ir=(0.229, 0.224, 0.225)):
        self.size = size
        self.mean_vis = mean_vis
        self.std_vis = std_vis
        self.mean_ir = mean_ir
        self.std_ir = std_ir

    def __call__(self, vis_img: Image.Image, ir_img: Image.Image, mask_pil: Image.Image):
        # Resize (bilinear for images, nearest for mask)
        vis_img = TF.resize(vis_img, self.size, interpolation=InterpolationMode.BILINEAR)
        ir_img  = TF.resize(ir_img,  self.size, interpolation=InterpolationMode.BILINEAR)
        mask_pil = TF.resize(mask_pil, self.size, interpolation=InterpolationMode.NEAREST)

        # Convert to tensors & normalize
        vis_t = TF.to_tensor(vis_img)
        ir_t  = TF.to_tensor(ir_img)
        vis_t = TF.normalize(vis_t, self.mean_vis, self.std_vis)
        ir_t  = TF.normalize(ir_t,  self.mean_ir,  self.std_ir)

        # Mask -> LongTensor (H, W)
        mask_np = np.array(mask_pil, dtype=np.uint8)
        target = torch.from_numpy(mask_np).long()
        return vis_t, ir_t, target


# --------- Dataset: MSRS parquet format ----------
class FusionParquetDataset(data.Dataset):
    """
    Dataset for fusion training/evaluation based on parquet files with columns:
      - image_name (str)
      - visible_image (bytes, RGB)
      - infrared_image (bytes, RGB)
      - question (str)
      - segmentation (NPY bytes; (H,W) uint8 mask with 0/1)

    Returns each sample as:
      vis_img:          FloatTensor (C, H, W)
      ir_img:           FloatTensor (C, H, W)
      target:           LongTensor (H, W)
      tensor_embedding: LongTensor (1, T)
      attention_mask:   LongTensor (1, T)
    """
    def __init__(self,
                 parquet_path_or_dir: str,
                 bert_tokenizer: str = "bert-base-uncased",
                 image_transforms: Any = None,
                 max_tokens: int = 20,
                 eval_mode: bool = False):
        super().__init__()
        self.df = _load_parquet_any(parquet_path_or_dir)
        self.tokenizer = BertTokenizer.from_pretrained(bert_tokenizer)
        self.max_tokens = max_tokens
        self.eval_mode = eval_mode
        self.image_transforms = image_transforms or PairedTwoImageTransform()

        # Pre-encode text to save time
        self._input_ids: List[torch.Tensor] = []
        self._attn_masks: List[torch.Tensor] = []
        for q in self.df["question"].tolist():
            ids = self.tokenizer.encode(text=str(q), add_special_tokens=True)[: self.max_tokens]
            attn = [1] * len(ids)
            if len(ids) < self.max_tokens:
                pad = self.max_tokens - len(ids)
                ids  = ids + [0] * pad
                attn = attn + [0] * pad
            self._input_ids.append(torch.tensor(ids, dtype=torch.long).unsqueeze(0))      # (1,T)
            self._attn_masks.append(torch.tensor(attn, dtype=torch.long).unsqueeze(0))    # (1,T)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index: int):
        row = self.df.iloc[index]

        # 1) Decode visible & infrared images
        vis_img = _pil_from_bytes(row["visible_image"])
        ir_img  = _pil_from_bytes(row["infrared_image"])

        # 2) Decode segmentation mask
        mask_np = _mask_from_npy_bytes(row["segmentation"])     # (H,W) uint8 in {0,1}
        mask_pil = Image.fromarray(mask_np, mode="P")

        # 3) Apply paired transform
        vis_t, ir_t, target = self.image_transforms(vis_img, ir_img, mask_pil)

        # 4) Pre-tokenized text
        tensor_embedding = self._input_ids[index]      # (1,T)
        attention_mask   = self._attn_masks[index]     # (1,T)

        return vis_t, ir_t, target, tensor_embedding, attention_mask
