import argparse
import sys
import torch

def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Configuration")

    ###### data.py ######
    parser.add_argument(
        "--root_dir", type=str, default="/nas/research/03-Neural_decoding",
        help="Path to the BIDS root."
    )
    parser.add_argument(
        "--fmri_dir", type=str, default="3-bids/derivatives",
        help="Path to the BIDS fmri."
    )
    parser.add_argument(
        "--fmri_detail_dir", type=str, default="beta_huggingface",
        choices=["b4_roi_zscore"],
        help="Path to the BIDS fmri_detail."
    )
    parser.add_argument(
        "--image_dir", type=str, default="4-image/beta",
        help="Path to the BIDS image."
    )
    parser.add_argument(
        "--seed",type=int,default=42,
    )
    parser.add_argument(
        '--mode', type=str, choices=['train', 'inference', 'evaluate'], default='inference',
        help="train, inference, evaluate 구분"
    )
    parser.add_argument(
        "--batch_size", type=int, default=80,
        help="Batch size can be increased by 10x if only training v2c and not diffusion prior",
    )
    parser.add_argument(
        "--prefetch_factor", type=int, default=10, choices=[2,4,6,8],
        help="한 프로세스에서 몇 개 처리할지",
    )
    parser.add_argument(
        "--num_workers", type=int, default=30, choices=[4,8,12,16,20],
        help="multi-processing in dataloader-메모리와 cpu개수에 맞게 조정",
    )
    parser.add_argument(
        "--num_epochs",type=int,default=270, choices=[3,240],
        help="epoch 개수",
    )
    parser.add_argument(
        "--is_shuffle",type=argparse.BooleanOptionalAction,default=False,
        help="is shuffle",
    )
    parser.add_argument(
        "--use_low_image",action=argparse.BooleanOptionalAction,default=True,
        help="embedding 사용 유무",
    )
    parser.add_argument(
        "--world_size",type=int,default=1,
        help="is shuffle",
    )
    parser.add_argument(
        "--rank",type=int,default=0,
        help="is shuffle",
    )
    ####################

    ###### mindeye1 ######
    parser.add_argument(
        "--device",type=str,default="cuda:3",
        help='device',
    )
    parser.add_argument(
        "--subj",type=int, default=1, choices=[1,2,5,7],
    )
    parser.add_argument(
        "--clip_size",type=int,default=768,
        help='clip embedding 크기',
    )
    parser.add_argument(
        "--token_size",type=int,default=257,
        help='vit patch 개수 + 1',
    )
    parser.add_argument(
        "--clip_variant",type=str,default="ViT-L/14",
        choices=["RN50", "ViT-L/14", "ViT-B/32", "RN50x64"],
        help='OpenAI clip variant',
    )
    parser.add_argument(
        "--norm_embs",action=argparse.BooleanOptionalAction,default=True,
        help="embedding 사용 유무",
    )
    parser.add_argument(
        "--hidden",action=argparse.BooleanOptionalAction,default=True,
        help="if True, CLIP embeddings will come from last hidden layer (e.g., 257x768 - Versatile Diffusion), rather than final layer",
    )
    parser.add_argument(
        "--cache_dir", type=str, default='/nas/research/03-Neural_decoding/5-mindeye_code/pretrained_cache',
        help="Where is cached Diffusion model; if not cached will download to this path",
    )
    ####################

    ###### optimizer ######
    parser.add_argument(
        "--optimizer",type=str,default='adamw',
    )
    ####################

    ###### scheduler ######
    parser.add_argument(
        "--max_lr",type=float,default=3e-4,
    )
    parser.add_argument(
        "--scheduler_type",type=str,default='cycle',
        choices=['cycle','linear'],
    )
    ####################

    ###### trainer ######
    parser.add_argument(
        "--mixup_pct",type=float,default=0.33,
        help="BiMixCo에서 SoftCLIP로 넘어가는 epoch",
    )
    parser.add_argument(
        "--prior_loss_coefficient",type=float,default=0.3,
        help="prior loss 계수",
    )
    parser.add_argument(
        "--code_dir", type=str, default="5-mindeye_code",
        help="Path to the code."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output",
        help="Path to the output."
    )
    parser.add_argument(
        "--model_name", type=str, default="mindeye1",
        help="모델 이름"
    )
    parser.add_argument(
        "--inference_batch_size",type=int,default=10,
        help="versatile inference batch size",
    )
    parser.add_argument(
        "--recons_per_sample", type=int, default=1,
        help= "한 frmi로 몇 개 sampling할 지"
    )
    parser.add_argument(
        "--num_inference_steps", type=int, default=20,
        help= "versatile inference step"
    )
    parser.add_argument(
        "--recon_name", type=str, default="mindeye1_recon",
        help="recon 캐시 이름"
    )
    parser.add_argument(
        "--metrics_name", type=str, default="mindeye1_metric",
        help="metric 결과 이름"
    )
    ####################

    # Jupyter 환경에서는 빈 리스트를 전달해야 실행이 됨
    if any("ipykernel_launcher" in arg for arg in sys.argv):
        args = parser.parse_args([])  
    else:
        args = parser.parse_args()

    return args

