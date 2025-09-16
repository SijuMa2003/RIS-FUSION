import torch.nn as nn
from .unet_fuser import UNetFuser


def _get_means_stds_from_args(args):
    """
    Retrieve normalization means and standard deviations for visible and infrared images.
    Defaults follow common ImageNet normalization if not provided in args.
    """
    mean_vis = getattr(args, "mean_vis", (0.485, 0.456, 0.406))
    std_vis  = getattr(args, "std_vis",  (0.229, 0.224, 0.225))
    mean_ir  = getattr(args, "mean_ir",  mean_vis)
    std_ir   = getattr(args, "std_ir",   std_vis)
    return mean_vis, std_vis, mean_ir, std_ir


def build_prefusion(name: str, args) -> nn.Module:
    """
    Build the pre-fusion model (only UNet version is kept).

    Parameters
    ----------
    name : str
        Model name, supports "unet" / "unet_fuser" (case-insensitive).
    args : Any
        Runtime configuration object; optional attributes include:
        - prefusion_base_ch: base channel size for UNet (default: 32)
        - mean_vis/std_vis/mean_ir/std_ir: normalization parameters
    """
    mean_vis, std_vis, mean_ir, std_ir = _get_means_stds_from_args(args)
    name = (name or "").lower()

    if name in ["unet", "unet_fuser"]:
        model = UNetFuser(
            in_ch=2,
            base_ch=getattr(args, "prefusion_base_ch", 32),
            mean_vis=mean_vis, std_vis=std_vis, mean_ir=mean_ir, std_ir=std_ir,
        )
    else:
        raise ValueError(
            f"Unknown prefusion model: {name}. "
            f"Only 'unet' is supported in this release."
        )

    return model