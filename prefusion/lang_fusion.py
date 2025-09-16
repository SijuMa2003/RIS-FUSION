# prefusion/modules/lang_fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextAdapter(nn.Module):
    """
    Project BERT tokens from [B, N, 768] to a target text dimension (tdim),
    followed by LayerNorm.
    """
    def __init__(self, in_dim: int = 768, tdim: int = 256):
        super().__init__()
        self.proj = nn.Linear(in_dim, tdim)
        self.norm = nn.LayerNorm(tdim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [B, N, 768]
        Returns:
            [B, N, tdim]
        """
        x = self.proj(tokens)
        x = self.norm(x)
        return x


class PixelWordAttention2D(nn.Module):
    """
    Pixel–word cross-attention: queries come from a feature map; keys/values come from text tokens.

    Shapes:
        x:      [B, C, H, W]
        tokens: [B, N, tdim]
        mask:   [B, N] with 1=valid, 0=padding

    Returns:
        context feature: [B, tdim, H, W]
    """
    def __init__(self, c: int, tdim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.scale = (tdim // heads) ** -0.5
        self.q_proj = nn.Conv2d(c, tdim, kernel_size=1)
        self.k_proj = nn.Linear(tdim, tdim)
        self.v_proj = nn.Linear(tdim, tdim)
        self.out_proj = nn.Conv2d(tdim, tdim, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        tdim = tokens.size(-1)

        # Q: [B, tdim, H, W] -> [B, heads, HW, dh]
        q = self.q_proj(x)  # [B, tdim, H, W]
        q = q.view(B, self.heads, tdim // self.heads, H * W)
        q = q.permute(0, 1, 3, 2)  # [B, h, HW, dh]

        # K/V: [B, N, tdim] -> [B, h, N, dh]
        k = self.k_proj(tokens)
        v = self.v_proj(tokens)
        k = k.view(B, -1, self.heads, tdim // self.heads).permute(0, 2, 1, 3)
        v = v.view(B, -1, self.heads, tdim // self.heads).permute(0, 2, 1, 3)

        # Attention: [B, h, HW, N]
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            # Broadcast mask [B, N] -> [B, 1, 1, N]
            attn = attn.masked_fill(mask[:, None, None, :] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)

        # Context: [B, h, HW, dh] -> [B, tdim, H, W]
        ctx = torch.matmul(attn, v)  # [B, h, HW, dh]
        ctx = ctx.permute(0, 1, 3, 2).contiguous().view(B, tdim, H, W)
        ctx = self.out_proj(ctx)
        return ctx


class LangGatedFusionBlock(nn.Module):
    """
    Language-guided fusion block:

      1) ctx = PixelWordAttention2D(x_for_q, tokens)   -> [B, tdim, H, W]
      2) α   = sigmoid(Conv(ctx))                      : gate to blend IR and VIS
      3) γ,β = Conv(ctx)                               : FiLM-style modulation
      4) out = (1 + γ_scale * γ) * ((1 − α) * vis + α * ir) + β

    Notes:
      - x_for_q can be concat([vis, ir]) or any intermediate feature used to generate Q.
      - c_in should match the channel count of x_for_q.
    """
    def __init__(self, c_in: int, tdim: int = 256, heads: int = 4, hidden: int = 128):
        super().__init__()
        self.pwam = PixelWordAttention2D(c_in, tdim, heads=heads)
        self.fuse_gate = nn.Sequential(
            nn.Conv2d(tdim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 1, kernel_size=1),
            nn.Sigmoid(),
        )
        self.film = nn.Sequential(
            nn.Conv2d(tdim, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 2, kernel_size=1),  # outputs [γ, β]
        )
        # keep modulation stable at initialization
        self.gamma_scale = 0.1

    def forward(
        self,
        feat_vis: torch.Tensor,
        feat_ir: torch.Tensor,
        x_for_q: torch.Tensor,
        tokens: torch.Tensor,
        token_mask: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_vis:   [B, C, H, W]
            feat_ir:    [B, C, H, W]
            x_for_q:    feature map used to generate queries (e.g., concat([feat_vis, feat_ir]))
            tokens:     [B, N, tdim]
            token_mask: [B, N] with 1=valid, 0=padding

        Returns:
            out:   fused feature map [B, C, H, W]
            alpha: fusion gate map    [B, 1, H, W]
        """
        ctx = self.pwam(x_for_q, tokens, token_mask)   # [B, tdim, H, W]
        alpha = self.fuse_gate(ctx)                    # [B, 1, H, W]
        fused = (1.0 - alpha) * feat_vis + alpha * feat_ir

        gb = self.film(ctx)                            # [B, 2, H, W]
        gamma, beta = torch.chunk(gb, 2, dim=1)

        out = (1.0 + self.gamma_scale * gamma) * fused + beta
        return out, alpha