from args import parse_args
from data import get_dataloader, sub1_train_dataset
from mindeye1 import get_model_highlevel
from optimizers import get_optimizer
from schedulers import get_scheduler
from trainer import train
import wandb

def main():
    # parse_args 정의
    args = parse_args()
    # data loader
    dataloader = get_dataloader(args)
    # model 정의
    models = get_model_highlevel(args) 
    model_bundle = {
        "clip": models["clip"],
        "diffusion_prior": models["diffusion_prior"],
        "vd_pipe": models["vd_pipe"], # inference에서만 사용
        "unet": models["unet"], # inference에서만 사용
        "vae": models["vae"], # inference에서만 사용
        "noise_scheduler": models["noise_scheduler"], # inference에서만 사용
    }
    # optimizer 정의
    optimizer = get_optimizer(args, model_bundle["diffusion_prior"])
    # scheduler 정의
    if args.mode == 'train':
        train_dataset = sub1_train_dataset(args)
        num_train = len(train_dataset)//3 # 저자는 같은 image 3번 본것을 1번으로 침
        lr_scheduler = get_scheduler(args, optimizer, num_train)

        wandb.init(project="neural_decoding",config=vars(args),name=f"run-{wandb.util.generate_id()}")

        # train 시작
        train(args, dataloader, model_bundle, optimizer, lr_scheduler)

    

if __name__ == "__main__":
    main()