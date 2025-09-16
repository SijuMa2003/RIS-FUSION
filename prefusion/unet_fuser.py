import torch
import torch.nn as nn
import torch.nn.functional as F

# Import shared language modules
from .lang_fusion import TextAdapter, PixelWordAttention2D, LangGatedFusionBlock


# === Color-space utilities (operate in [0, 1] float range) ===
def rgb_to_ycbcr(rgb: torch.Tensor):
    """Convert RGB to YCbCr (channel-first tensors in [0, 1])."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    y  = 0.299 * r + 0.587 * g + 0.114 * b
    cb = 0.5 - 0.168736 * r - 0.331264 * g + 0.5 * b
    cr = 0.5 + 0.5 * r - 0.418688 * g - 0.081312 * b
    return y, cb, cr

def ycbcr_to_rgb(y: torch.Tensor, cb: torch.Tensor, cr: torch.Tensor) -> torch.Tensor:
    """Convert YCbCr back to RGB and clamp to [0, 1]."""
    r = y + 1.402 * (cr - 0.5)
    g = y - 0.344136 * (cb - 0.5) - 0.714136 * (cr - 0.5)
    b = y + 1.772 * (cb - 0.5)
    rgb = torch.cat([r, g, b], dim=1)
    return torch.clamp(rgb, 0.0, 1.0)

def rgb_to_gray(rgb: torch.Tensor) -> torch.Tensor:
    """Convert RGB to single-channel grayscale (luma approximation)."""
    r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


# === Generic U-Net components ===
class DoubleConv(nn.Module):
    """Two consecutive 3x3 convs + BN + ReLU."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class Down(nn.Module):
    """Downsampling block: MaxPool -> DoubleConv."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_ch, out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pool(x))

class Up(nn.Module):
    """
    Upsampling block: ConvTranspose2d (x2) + concat skip + DoubleConv.
    """
    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        self.conv = DoubleConv(in_ch // 2 + skip_ch, out_ch)

    def _align_to(self, x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        """Center-pad/crop x to match spatial size of ref."""
        Ht, Wt = x.size(-2), x.size(-1)
        Hr, Wr = ref.size(-2), ref.size(-1)

        dh = Hr - Ht
        if dh > 0:
            top = dh // 2; bottom = dh - top
            x = F.pad(x, (0, 0, top, bottom))
        elif dh < 0:
            crop = -dh; top = crop // 2; bottom = crop - top
            x = x[..., top:Ht-bottom, :]

        dw = Wr - Wt
        if dw > 0:
            left = dw // 2; right = dw - left
            x = F.pad(x, (left, right, 0, 0))
        elif dw < 0:
            crop = -dw; left = crop // 2; right = crop - left
            x = x[..., :, left:Wt-right]

        return x

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = self._align_to(x, skip)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


# === Two-stream encoder + language-injected U-Net ===
class UNet2StreamLang(nn.Module):
    """
    Two encoders (Y and IRy). Language-guided fusion at /4 and /8 scales,
    simple averaging at lower scales. Standard U-shaped decoder.
    """
    def __init__(self, base_ch: int = 32, tdim: int = 256, heads: int = 4,
                 use_levels=(3, 4)):
        super().__init__()
        c = base_ch
        # Y branch
        self.y_inc = DoubleConv(1, c)
        self.y_d1  = Down(c, c * 2)
        self.y_d2  = Down(c * 2, c * 4)
        self.y_d3  = Down(c * 4, c * 8)
        # IR branch
        self.i_inc = DoubleConv(1, c)
        self.i_d1  = Down(c, c * 2)
        self.i_d2  = Down(c * 2, c * 4)
        self.i_d3  = Down(c * 4, c * 8)
        # Language-gated fusion (note: c_in is channels of concat([feat_y, feat_ir]))
        self.use_l3 = 3 in set(use_levels)  # /4
        self.use_l4 = 4 in set(use_levels)  # /8
        if self.use_l3:
            # y3/i3: c*4 each -> concat: c*8
            self.lang_l3 = LangGatedFusionBlock(c_in=c * 8, tdim=tdim, heads=heads)
        if self.use_l4:
            # y4/i4: c*8 each -> concat: c*16
            self.lang_l4 = LangGatedFusionBlock(c_in=c * 16, tdim=tdim, heads=heads)
        
        # Decoder
        self.u3  = Up(in_ch=c * 8,  skip_ch=c * 4, out_ch=c * 4)
        self.u2  = Up(in_ch=c * 4,  skip_ch=c * 2, out_ch=c * 2)
        self.u1  = Up(in_ch=c * 2,  skip_ch=c,     out_ch=c)
        self.outc = nn.Conv2d(c, 1, kernel_size=1)

        # Text adapter
        self.txt = TextAdapter(in_dim=768, tdim=tdim)

    def forward(
        self,
        Y: torch.Tensor,
        IRy: torch.Tensor,
        text_feats: torch.Tensor = None,
        text_mask: torch.Tensor = None
    ) -> torch.Tensor:
        tokens = self.txt(text_feats) if (text_feats is not None) else None

        # encoders
        y1 = self.y_inc(Y)
        y2 = self.y_d1(y1)
        y3 = self.y_d2(y2)
        y4 = self.y_d3(y3)

        i1 = self.i_inc(IRy)
        i2 = self.i_d1(i1)
        i3 = self.i_d2(i2)
        i4 = self.i_d3(i3)

        # fusion (pass x_for_q explicitly as concat([feat_y, feat_ir]))
        if self.use_l3:
            xq3 = torch.cat([y3, i3], dim=1)
            f3, _ = self.lang_l3(y3, i3, x_for_q=xq3, tokens=tokens, token_mask=text_mask)
        else:
            f3 = 0.5 * (y3 + i3)

        if self.use_l4:
            xq4 = torch.cat([y4, i4], dim=1)
            f4, _ = self.lang_l4(y4, i4, x_for_q=xq4, tokens=tokens, token_mask=text_mask)
        else:
            f4 = 0.5 * (y4 + i4)

        f2 = 0.5 * (y2 + i2)
        f1 = 0.5 * (y1 + i1)

        # decoder
        x  = self.u3(f4, f3)
        x  = self.u2(x,  f2)
        x  = self.u1(x,  f1)
        y  = self.outc(x)
        return torch.sigmoid(y)


class UNetFuser(nn.Module):
    """
    Two-stream UNet-based fuser with optional language guidance.
    Workflow:
      1) De-normalize inputs to [0, 1]
      2) Convert VIS RGB -> (Y, Cb, Cr), IR RGB -> grayscale IRy
      3) Run two-stream UNet with language-guided fusion
      4) Replace Y with fused Y_fused, then YCbCr -> RGB
      5) Re-normalize to match downstream models
    """
    def __init__(self, in_ch: int = 2, base_ch: int = 32,
                 mean_vis=(0.485, 0.456, 0.406), std_vis=(0.229, 0.224, 0.225),
                 mean_ir=(0.485, 0.456, 0.406),  std_ir=(0.229, 0.224, 0.225),
                 tdim: int = 256, heads: int = 4, inject_levels=(3, 4)):
        super().__init__()
        self.net = UNet2StreamLang(base_ch=base_ch, tdim=tdim, heads=heads, use_levels=inject_levels)

        self.register_buffer("mean_vis", torch.tensor(mean_vis).view(1, 3, 1, 1))
        self.register_buffer("std_vis",  torch.tensor(std_vis ).view(1, 3, 1, 1))
        self.register_buffer("mean_ir",  torch.tensor(mean_ir ).view(1, 3, 1, 1))
        self.register_buffer("std_ir",   torch.tensor(std_ir  ).view(1, 3, 1, 1))

    def _denorm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return torch.clamp(x * std + mean, 0.0, 1.0)

    def _norm(self, x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        return (x - mean) / std

    @torch.no_grad()
    def _ensure_size(self, y: torch.Tensor, ref_hw) -> torch.Tensor:
        if y.shape[-2:] != ref_hw:
            return F.interpolate(y, size=ref_hw, mode="bilinear", align_corners=False)
        return y

    def forward(
        self,
        vis: torch.Tensor,
        ir: torch.Tensor,
        text_feats: torch.Tensor = None,
        text_mask: torch.Tensor = None
    ) -> torch.Tensor:
        # De-normalize
        vis01 = self._denorm(vis, self.mean_vis, self.std_vis)
        ir01  = self._denorm(ir,  self.mean_ir,  self.std_ir)

        # RGB -> YCbCr (fuse Y only)
        Y, Cb, Cr = rgb_to_ycbcr(vis01)
        IRy = rgb_to_gray(ir01)

        # Fusion
        Yf = self.net(Y, IRy, text_feats=text_feats, text_mask=text_mask)

        # Align to original Y size if needed
        if Yf.shape[-2:] != Y.shape[-2:]:
            Yf = F.interpolate(Yf, size=Y.shape[-2:], mode="bilinear", align_corners=False)

        # Recompose RGB and re-normalize
        rgb_f = ycbcr_to_rgb(Yf, Cb, Cr)
        out = self._norm(rgb_f, self.mean_vis, self.std_vis)
        return out
