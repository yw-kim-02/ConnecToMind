import os
import gc

import torch
import wandb

from args import parse_args
from data import get_dataloader, sub1_train_dataset
from mindeye1 import get_model_highlevel
from optimizers import get_optimizer
from schedulers import get_scheduler
from trainer import train
from utils import get_unique_path

def main():
    # parse_args 정의
    args = parse_args()

    # data loader
    dataloader = get_dataloader(args)

    # model 정의
    models = get_model_highlevel(args) 
    model_bundle = {
        "clip": models["clip"].to(args.device),
        "diffusion_prior": models["diffusion_prior"].to(args.device),
        # "vd_pipe": models["vd_pipe"].to(args.device), # inference에서만 사용
        # "unet": models["unet"].to(args.device), # inference에서만 사용
        # "vae": models["vae"].to(args.device), # inference에서만 사용
        # "noise_scheduler": models["noise_scheduler"], # inference에서만 사용
    }

    if args.mode == 'train':
        # optimizer 정의
        optimizer = get_optimizer(args, model_bundle["diffusion_prior"])

        # scheduler 정의(train만 함)
        train_dataset = sub1_train_dataset(args)
        num_train = len(train_dataset)//3 # 저자는 같은 image 3번 본것을 1번으로 침
        lr_scheduler = get_scheduler(args, optimizer, num_train)

        # wandb 적용
        wandb.login() # login
        wandb.init(project="neural_decoding", name=f"run-{wandb.util.generate_id()}") # init
        wandb.config = vars(args) # aparse_args()의 내용 그대로 config로 주기

        # train 시작
        output_model = train(args, dataloader, model_bundle, optimizer, lr_scheduler)

        # model 저장
        output_path = os.path.join(args.root_dir ,args.output_dir, args.model_name)
        output_path = get_unique_path(output_path)
        torch.save(output_model.state_dict(), output_path)

if __name__ == "__main__":
    main()