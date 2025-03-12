import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")

    ###### data.py ######
    parser.add_argument(
        "--root_dir", type=str, default="/nas/research/03-Neural_decoding",
        help="Path to the BIDS directory."
    )
    parser.add_argument(
        "--seed",type=int,default=42,
    )
    parser.add_argument(
        "--batch_size", type=int, default=32,
        help="Batch size can be increased by 10x if only training v2c and not diffusion prior",
    )
    parser.add_argument(
        "--num_workers", type=str, default=4,
        help="multi-processing in dataloader",
    )
    parser.add_argument(
        "--is_shuffle",type=int,default=True,
        help="is shuffle",
    )
    ####################



    parser.add_argument(
        "--model_name", type=str, default="testing",
        help="name of model, used for ckpt saving and wandb logging (if enabled)",
    )
    parser.add_argument(
        "--data_path", type=str, default="/fsx/proj-medarc/fmri/natural-scenes-dataset",
        help="Path to where NSD data is stored / where to download it to",
    )
    parser.add_argument(
        "--subj",type=int, default=1, choices=[1,2,5,7],
    )
    parser.add_argument(
        "--hidden",action=argparse.BooleanOptionalAction,default=True,
        help="if True, CLIP embeddings will come from last hidden layer (e.g., 257x768 - Versatile Diffusion), rather than final layer",
    )
    parser.add_argument(
        "--clip_variant",type=str,default="ViT-L/14",choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
        help='OpenAI clip variant',
    )
    parser.add_argument(
        "--wandb_log",action=argparse.BooleanOptionalAction,default=False,
        help="whether to log to wandb",
    )
    parser.add_argument(
        "--resume_from_ckpt",action=argparse.BooleanOptionalAction,default=False,
        help="if not using wandb and want to resume from a ckpt",
    )
    parser.add_argument(
        "--wandb_project",type=str,default="stability",
        help="wandb project name",
    )
    parser.add_argument(
        "--mixup_pct",type=float,default=.33,
        help="proportion of way through training when to switch from BiMixCo to SoftCLIP",
    )
    parser.add_argument(
        "--norm_embs",action=argparse.BooleanOptionalAction,default=True,
        help="Do l2-norming of CLIP embeddings",
    )
    parser.add_argument(
        "--use_image_aug",action=argparse.BooleanOptionalAction,default=True,
        help="whether to use image augmentation",
    )
    parser.add_argument(
        "--num_epochs",type=int,default=240,
        help="number of epochs of training",
    )
    parser.add_argument(
        "--prior",action=argparse.BooleanOptionalAction,default=True,
        help="if False, will only use CLIP loss and ignore diffusion prior",
    )
    parser.add_argument(
        "--v2c",action=argparse.BooleanOptionalAction,default=True,
        help="if False, will only use diffusion prior loss",
    )
    parser.add_argument(
        "--plot_umap",action=argparse.BooleanOptionalAction,default=False,
        help="Plot UMAP plots alongside reconstructions",
    )
    parser.add_argument(
        "--lr_scheduler_type",type=str,default='cycle',choices=['cycle','linear'],
    )
    parser.add_argument(
        "--ckpt_saving",action=argparse.BooleanOptionalAction,default=True,
    )
    parser.add_argument(
        "--ckpt_interval",type=int,default=5,
        help="save backup ckpt and reconstruct every x epochs",
    )
    parser.add_argument(
        "--save_at_end",action=argparse.BooleanOptionalAction,default=False,
        help="if True, saves best.ckpt at end of training. if False and ckpt_saving==True, will save best.ckpt whenever epoch shows best validation score",
    )

    parser.add_argument(
        "--max_lr",type=float,default=3e-4,
    )
    parser.add_argument(
        "--n_samples_save",type=int,default=0,choices=[0,1],
        help="Number of reconstructions for monitoring progress, 0 will speed up training",
    )
    parser.add_argument(
        "--use_projector",action=argparse.BooleanOptionalAction,default=True,
        help="Additional MLP after the main MLP so model can separately learn a way to minimize NCE from prior loss (BYOL)",
    )
    parser.add_argument(
        "--vd_cache_dir", type=str, default='/fsx/proj-medarc/fmri/cache/models--shi-labs--versatile-diffusion/snapshots/2926f8e11ea526b562cd592b099fcf9c2985d0b7',
        help="Where is cached Versatile Diffusion model; if not cached will download to this path",
    )

    # Jupyter 환경에서는 빈 리스트 전달해야 실행이 됨
    if any("ipykernel_launcher" in arg for arg in sys.argv):
        args = parser.parse_args([])  
    else:
        args = parser.parse_args()

    return args

