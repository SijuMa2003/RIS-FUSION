# losses/fusion_losses.py — SSIM + Sobel edges + joint gradient (with optional MSE)
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------- Utilities ----------
def _rgb_to_y(rgb01: torch.Tensor) -> torch.Tensor:
    """Convert RGB image [B,3,H,W] in [0,1] to luminance Y channel [B,1,H,W]."""
    r, g, b = rgb01[:, 0:1], rgb01[:, 1:2], rgb01[:, 2:3]
    return 0.299 * r + 0.587 * g + 0.114 * b


def _denorm_from_prefusion(x: torch.Tensor, prefusion_model, domain: str) -> torch.Tensor:
    """
    De-normalize tensor x from standardized domain to [0,1].

    Args:
        x (Tensor): [B,3,H,W], normalized input.
        prefusion_model: Prefusion model holding registered mean/std buffers.
        domain (str): Either "vis" or "ir".

    Returns:
        Tensor: De-normalized [B,3,H,W] in [0,1].
    """
    assert domain in ("vis", "ir")
    mean = getattr(prefusion_model, f"mean_{domain}").to(x.device)  # [1,3,1,1]
    std = getattr(prefusion_model, f"std_{domain}").to(x.device)    # [1,3,1,1]
    return torch.clamp(x * std + mean, 0.0, 1.0)


def _create_gaussian_window(window_size: int, sigma: float, device, dtype):
    """Create 2D Gaussian kernel [1,1,w,w]."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2 * sigma * sigma))
    g = g / g.sum()
    window_1d = g.unsqueeze(1)
    return (window_1d @ window_1d.t()).unsqueeze(0).unsqueeze(0)


def ssim_1ch(x: torch.Tensor, y: torch.Tensor, window_size: int = 11, sigma: float = 1.5) -> torch.Tensor:
    """
    Compute SSIM between two single-channel images.

    Args:
        x, y (Tensor): [B,1,H,W] in [0,1].
        window_size (int): Gaussian window size.
        sigma (float): Gaussian sigma.

    Returns:
        Tensor: Mean SSIM scalar.
    """
    assert x.shape == y.shape and x.dim() == 4 and x.size(1) == 1
    device, dtype = x.device, x.dtype
    window = _create_gaussian_window(window_size, sigma, device, dtype)
    padding = window_size // 2

    mu_x, mu_y = F.conv2d(x, window, padding=padding), F.conv2d(y, window, padding=padding)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sigma_x2 = F.conv2d(x * x, window, padding=padding) - mu_x2
    sigma_y2 = F.conv2d(y * y, window, padding=padding) - mu_y2
    sigma_xy = F.conv2d(x * y, window, padding=padding) - mu_xy

    C1, C2 = 0.01**2, 0.03**2
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean()


# ---------- Sobel gradient operator ----------
class Sobelxy(nn.Module):
    """Compute Sobel gradients |Gx|, |Gy| for single-channel images [B,1,H,W]."""

    def __init__(self, device):
        super().__init__()
        kernelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
        kernely = [[1, 2, 1], [0, 0, 0], [-1, -2, -1]]
        kx = torch.tensor(kernelx, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ky = torch.tensor(kernely, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(kx, requires_grad=False).to(device=device)
        self.weighty = nn.Parameter(ky, requires_grad=False).to(device=device)

    def forward(self, x: torch.Tensor):
        sobelx = F.conv2d(x, self.weightx, padding=1)
        sobely = F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx), torch.abs(sobely)


# ---------- Fusion loss ----------
class FusionLoss(nn.Module):
    """
    Prefusion loss combining SSIM, MSE, Sobel edge consistency, and joint gradient terms.

    Loss = 
        w_ssim_vis * (1 - SSIM(Yf, Yvis)) +
        w_ssim_ir  * (1 - SSIM(Yf, IRy)) +
        w_mse_vis  * MSE(Yf, Yvis) +
        w_mse_ir   * MSE(Yf, IRy) +
        w_sobel_vis * (|Gx(Yf)-Gx(Yvis)|_1 + |Gy(Yf)-Gy(Yvis)|_1) +
        w_sobel_ir  * (|Gx(Yf)-Gx(IRy)|_1  + |Gy(Yf)-Gy(IRy)|_1) +
        w_grad * (|Gx(Yf)-max(Gx(Yvis),Gx(IRy))|_1 + |Gy(Yf)-max(Gy(Yvis),Gy(IRy))|_1)
    """

    def __init__(self, args, device, prefusion_model):
        super().__init__()
        # loss weights (can be overridden via args)
        self.w_ssim_vis = float(getattr(args, "w_ssim_vis", 1.0))
        self.w_ssim_ir = float(getattr(args, "w_ssim_ir", 0.0))
        self.w_mse_ir = float(getattr(args, "w_mse_ir", 1.0))
        self.w_mse_vis = float(getattr(args, "w_mse_vis", 0.0))
        self.w_sobel_vis = float(getattr(args, "w_sobel_vis", 1.0))
        self.w_sobel_ir = float(getattr(args, "w_sobel_ir", 1.0))
        self.w_grad = float(getattr(args, "w_grad", 1.0))
        self.lambda_pf = float(getattr(args, "lambda_prefusion", 1.0))

        # SSIM config
        self.ssim_win = int(getattr(args, "ssim_window", 11))
        self.ssim_sigma = float(getattr(args, "ssim_sigma", 1.5))

        # Prefusion model (for de-normalization)
        self.prefusion_model = prefusion_model

        # Sobel operator
        self.sobel = Sobelxy(device)

    def prefusion_loss(self, fused_std, vis_std, ir_std):
        """
        Compute prefusion loss.

        Args:
            fused_std, vis_std, ir_std (Tensor): [B,3,H,W] normalized images.
        Returns:
            Tuple: (loss scalar, stats dict)
        """
        # de-normalize to [0,1], then extract Y channel
        fused01 = _denorm_from_prefusion(fused_std, self.prefusion_model, "vis")
        vis01 = _denorm_from_prefusion(vis_std, self.prefusion_model, "vis")
        ir01 = _denorm_from_prefusion(ir_std, self.prefusion_model, "ir")

        Yf, Yvis, IRy = _rgb_to_y(fused01), _rgb_to_y(vis01), _rgb_to_y(ir01)

        # SSIM loss
        ssim_vis = ssim_1ch(Yf, Yvis, self.ssim_win, self.ssim_sigma)
        ssim_ir = ssim_1ch(Yf, IRy, self.ssim_win, self.ssim_sigma)
        loss_ssim = self.w_ssim_vis * (1 - ssim_vis) + self.w_ssim_ir * (1 - ssim_ir)

        # MSE loss
        mse_vis, mse_ir = F.mse_loss(Yf, Yvis), F.mse_loss(Yf, IRy)
        loss_mse = self.w_mse_vis * mse_vis + self.w_mse_ir * mse_ir

        # Sobel edge losses
        gx_f, gy_f = self.sobel(Yf)
        gx_vis, gy_vis = self.sobel(Yvis)
        gx_ir, gy_ir = self.sobel(IRy)

        sobel_vis = F.l1_loss(gx_f, gx_vis) + F.l1_loss(gy_f, gy_vis)
        sobel_ir = F.l1_loss(gx_f, gx_ir) + F.l1_loss(gy_f, gy_ir)
        loss_sobel = self.w_sobel_vis * sobel_vis + self.w_sobel_ir * sobel_ir

        # Joint gradient loss
        grad_joint_x, grad_joint_y = torch.maximum(gx_vis, gx_ir), torch.maximum(gy_vis, gy_ir)
        loss_grad = self.w_grad * (F.l1_loss(gx_f, grad_joint_x) + F.l1_loss(gy_f, grad_joint_y))

        # total loss
        pf_loss = loss_ssim + loss_sobel + loss_grad + loss_mse

        stats = {
            "pf_total": float(pf_loss.detach()),
            "pf_ssim_vis": float(ssim_vis.detach()),
            "pf_ssim_ir": float(ssim_ir.detach()),
            "pf_mse_vis": float(mse_vis.detach()),
            "pf_mse_ir": float(mse_ir.detach()),
            "pf_sobel_vis": float(sobel_vis.detach()),
            "pf_sobel_ir": float(sobel_ir.detach()),
            "pf_grad": float(loss_grad.detach()),
        }
        return pf_loss, stats


def build_loss(args, device, prefusion_model) -> FusionLoss:
    """Factory method for FusionLoss, keeping compatibility with existing code."""
    return FusionLoss(args=args, device=device, prefusion_model=prefusion_model)
