import os
import gc

import torch
import wandb

from args import parse_args
from data import get_dataloader, sub1_train_dataset
from mindeye1 import get_model_highlevel
from optimizers import get_optimizer
from schedulers import get_scheduler
from metrics import get_metric
from trainer import train, evaluate
from utils import get_unique_path

def main():
    # parse_args 정의
    args = parse_args()

    #### train ####
    # data loader
    train_data = get_dataloader(args)

    # model 정의
    models = get_model_highlevel(args) 
    model_bundle = {
        "clip": models["clip"].to(args.device),
        "diffusion_prior": models["diffusion_prior"].to(args.device),
    }

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
    output_model = train(args, train_data, model_bundle, optimizer, lr_scheduler)

    # model 저장
    output_path = os.path.join(args.root_dir ,args.output_dir, args.model_name)
    output_path = get_unique_path(output_path)
    torch.save(output_model.state_dict(), output_path)

    # gpu에서 train 비우기
    del train_data, train_dataset, optimizer, lr_scheduler, output_model, model_bundle
    gc.collect()
    torch.cuda.empty_cache()

    #### evalutate ####
    setattr(args, 'mode', 'test')
    test_data = get_dataloader(args)

    model_bundle = {
        "clip": models["clip"].to(args.device),
        "diffusion_prior": models["diffusion_prior"].to(args.device),
        "vd_pipe": models["vd_pipe"].to(args.device), # inference에서만 사용
        "unet": models["unet"].to(args.device), # inference에서만 사용
        "vae": models["vae"].to(args.device), # inference에서만 사용
        "noise_scheduler": models["noise_scheduler"], # inference에서만 사용
    }

    metrics = get_metric(args)
    metric_bundle = {
        "pixcorr": metric_bundle["pixcorr"],
        "ssim": metric_bundle["ssim"],
        "clip": {
            "model": metric_bundle["clip"]["model"].to(args.device),
            "preprocess": metric_bundle["clip"]["preprocess"],
            "metric_fn": metric_bundle["clip"]["metric_fn"],
        },
        "alexnet2": {
            "model": metric_bundle["alexnet2"]["model"].to(args.device),
            "preprocess": metric_bundle["alexnet2"]["preprocess"],
            "layer": metric_bundle["alexnet2"]["layer"],
            "metric_fn": metric_bundle["alexnet2"]["metric_fn"],
        },
        "alexnet5": {
            "model": metric_bundle["alexnet5"]["model"].to(args.device),
            "preprocess": metric_bundle["alexnet5"]["preprocess"],
            "layer": metric_bundle["alexnet5"]["layer"],
            "metric_fn": metric_bundle["alexnet5"]["metric_fn"],
        },
        "inception": {
            "model": metric_bundle["inception"]["model"].to(args.device),
            "preprocess": metric_bundle["inception"]["preprocess"],
            "metric_fn": metric_bundle["inception"]["metric_fn"],
        },
    }

    result_values = evaluate(args, test_data, model_bundle, output_path, metric_bundle)

    # metric 저장
    txt_path = os.path.join(args.root_dir, args.output_dir, "evaluation_results.txt")
    with open(txt_path, "w") as f:
        for name, score in result_values.items():
            f.write(f"{name}: {score:.4f}\n")

if __name__ == "__main__":
    main()