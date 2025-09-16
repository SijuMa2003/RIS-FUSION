import argparse


def get_parser():
    """
    Argument parser for LAVT training and evaluation.
    Includes dataset, model, optimization, and pre-fusion settings.
    """
    parser = argparse.ArgumentParser(description="LAVT training and testing")

    # ---------------- Optimization ----------------
    parser.add_argument("--amsgrad", action="store_true",
                        help="Use AMSGrad in Adam/AdamW optimizer.")
    parser.add_argument("-b", "--batch-size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--epochs", default=40, type=int, metavar="N",
                        help="Number of total epochs to run.")
    parser.add_argument("--lr", default=5e-5, type=float,
                        help="Initial learning rate.")
    parser.add_argument("--wd", "--weight-decay", dest="weight_decay",
                        default=1e-2, type=float, metavar="W",
                        help="Weight decay.")
    parser.add_argument("--lr_seg", type=float, default=None,
                        help="Learning rate for segmentation optimizer.")
    parser.add_argument("--lr_pf", type=float, default=None,
                        help="Learning rate for pre-fusion optimizer.")
    parser.add_argument("--wd_seg", type=float, default=None,
                        help="Weight decay for segmentation optimizer.")
    parser.add_argument("--wd_pf", type=float, default=None,
                        help="Weight decay for pre-fusion optimizer.")
    parser.add_argument("--fusion_drop", default=0.0, type=float,
                        help="Dropout rate for PWAMs.")

    # ---------------- Model ----------------
    parser.add_argument("--model", default="lavt", help="Model type: lavt, lavt_one.")
    parser.add_argument("--model_id", default="lavt", help="Identifier name for model.")
    parser.add_argument("--pretrained_swin_weights",
                        default="./pretrained_weights/swin_base_patch4_window12_384_22k.pth",
                        help="Path to pre-trained Swin Transformer weights.")
    parser.add_argument("--swin_type", default="base",
                        help="Variant of Swin Transformer: tiny, small, base, or large.")
    parser.add_argument("--window12", action="store_true",
                        help="Initialize Swin with window size 12 instead of 7. "
                             "Training infers window size from pre-trained weights filename.")
    parser.add_argument("--init_from_lavt_one", type=str,
                        default="./pretrained_weights/lavt_one_8_cards_ImgNet22KPre_swin-base-window12_refcoco+_adamw_b32lr0.00005wd1e-2_E40.pth",
                        help="Path to a pretrained lavt_one checkpoint (.pth/.pt).")
    parser.add_argument("--load_lavt_head", action="store_true",
                        help="Also load segmentation head if shapes match. Default: only backbone/decoder.")

    # ---------------- Pre-fusion ----------------
    parser.add_argument("--prefusion_model", type=str, default="unet_fuser",
                        choices=["unet_fuser", "resnet_fuser"],
                        help="Pre-fusion model type.")
    parser.add_argument("--ck_prefusion", type=str, default="",
                        help="(Optional) Path to pre-trained pre-fusion weights.")
    parser.add_argument("--prefusion_base_ch", type=int, default=32,
                        help="Base channels for pre-fusion network (UNet: 32; ResNet: 64).")
    parser.add_argument("--prefusion_blocks", type=int, default=6,
                        help="Number of residual blocks in ResNet pre-fusion (only for resnet_fuser).")
    parser.add_argument("--prefusion_debug", action="store_true",
                        help="Print pre-fusion model architecture and layer shapes.")
    parser.add_argument("--prefusion_dryrun", action="store_true",
                        help="Run a dummy forward pass to catch shape errors early.")

    # ---------------- Dataset ----------------
    parser.add_argument("--dataset", default="refcoco",
                        help="Dataset: refcoco, refcoco+, or refcocog.")
    parser.add_argument("--train_parquet", type=str, required=True,
                        help="Path to training parquet file/dir "
                             "(columns: image_name/visible_image/infrared_image/question/segmentation).")
    parser.add_argument("--val_parquet", type=str, required=True,
                        help="Path to validation parquet file/dir.")
    parser.add_argument("--refer_data_root", default="./refer/data/",
                        help="Root directory for REFER dataset.")
    parser.add_argument("--split", default="test", help="Split used during testing.")
    parser.add_argument("--splitBy", default="unc",
                        help="For G-Ref (RefCOCOg), set to 'umd' or 'google'.")

    # ---------------- BERT ----------------
    parser.add_argument("--bert_tokenizer", default="bert-base-uncased",
                        help="Name of the BERT tokenizer.")
    parser.add_argument("--ck_bert", default="bert-base-uncased",
                        help="Path to pre-trained BERT weights.")

    # ---------------- DDP / Hardware ----------------
    parser.add_argument("--device", default="cuda:0",
                        help="Device to use (only for single-machine testing).")
    parser.add_argument("--local_rank", type=int,
                        help="Local rank for DistributedDataParallel.")
    parser.add_argument("--ddp_trained_weights", action="store_true",
                        help="Specify if loading weights trained with DDP.")
    parser.add_argument("-j", "--workers", default=8, type=int, metavar="N",
                        help="Number of data loading workers.")
    parser.add_argument("--pin_mem", action="store_true",
                        help="Pin memory in dataloader.")

    # ---------------- Logging & IO ----------------
    parser.add_argument("--output-dir", default="./checkpoints/",
                        help="Directory to save checkpoints.")
    parser.add_argument("--resume", default="", help="Resume from checkpoint path.")
    parser.add_argument("--print-freq", default=10, type=int,
                        help="Print frequency during training.")
    parser.add_argument("--eval_vis_dir", default="./eval_vis", type=str,
                        help="Directory to save visualization results during evaluation.")

    # ---------------- Loss weights ----------------
    parser.add_argument("--lambda_prefusion", type=float, default=2.0)
    parser.add_argument("--w_sobel_vis", type=float, default=0.0)
    parser.add_argument("--w_sobel_ir", type=float, default=0.0)
    parser.add_argument("--w_grad", type=float, default=0.0)
    parser.add_argument("--w_ssim_vis", type=float, default=1.0)
    parser.add_argument("--w_mse_vis", type=float, default=0.0)
    parser.add_argument("--w_ssim_ir", type=float, default=0.0)
    parser.add_argument("--w_mse_ir", type=float, default=2.0)
    parser.add_argument("--ssim_window", type=int, default=11)
    parser.add_argument("--ssim_sigma", type=float, default=1.5)

    # ---------------- Misc ----------------
    parser.add_argument("--img_size", default=480, type=int,
                        help="Input image size.")
    parser.add_argument("--mha", default="",
                        help="PWAM heads config string, e.g., 4-4-4-4 for 4 stages.")
    parser.add_argument("--use_y_for_psnr",
                        type=lambda x: str(x).lower() in ["true", "1", "yes"],
                        default=True,
                        help="Whether to compute PSNR using Y channel only.")

    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
