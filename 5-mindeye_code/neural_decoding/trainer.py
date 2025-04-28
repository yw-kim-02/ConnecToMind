import os
import gc
import time
from tqdm import tqdm
import wandb

import numpy as np
import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.profiler import profile, record_function, ProfilerActivity

from utils import img_augment, mixup, mixco_nce_loss, cosine_anneal, soft_clip_loss, topk, batchwise_cosine_similarity, get_unique_path

def train(args, data, models, optimizer, lr_scheduler):

    device = args.device
    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    clip_size = args.clip_size
    prior_loss_coefficient = args.prior_loss_coefficient

    scaler = GradScaler() # autocast scaler 인스턴스 생성
    
    # model 정의
    clip_extractor = models["clip"]
    diffusion_prior = models["diffusion_prior"]
    optimizer = optimizer
    lr_scheduler = lr_scheduler

    # log list
    losses, lrs = [], []

    progress_bar = tqdm(range(0,num_epochs), ncols=1200)
    for epoch in progress_bar:
        
        diffusion_prior.train()

        # metric
        sims_base = 0.0 # cosinesimilarity[(fMRI → CLIP), (image → CLIP)]의 누적합 -> 평균 구할 때 쓰임
        fwd_percent_correct = 0.0 # forward prediction이 정답과의 cosinesimilarity가 가장 높으면 1, 아니면 0 -> 비율의 누적합 -> 평균 구할 때 쓰임
        bwd_percent_correct = 0.0 # backward prediction이 정답과의 cosinesimilarity가 가장 높으면 1, 아니면 0 -> 비율의 누적합 -> 평균 구할 때 쓰임
        loss_nce_sum = 0.0 # Negative Contrastive Estimation loss의 누적합 -> 평균 구할 때 쓰임
        loss_prior_sum = 0.0 # prior loss의 누적합 -> 평균 구할 때 쓰임

        for index, (fmri_vol, image) in enumerate(data): # enumerate: index와 값을 같이 반환
            # gradient 초기화
            optimizer.zero_grad() 

            # data gpu 올리기 - amp 사용
            fmri_vol = fmri_vol.to(device, non_blocking=True)   # fMRI -> GPU
            image = image.to(device, non_blocking=True)         # Image -> GPU
            
            # image augmentation
            image = img_augment(image)

            # epoch의 1/3 지점 까지만 mixup 사용
            if epoch < int(mixup_pct * num_epochs):
                fmri_vol, perm, betas, select = mixup(fmri_vol)
            
            with autocast():
                # target 정의
                clip_target = clip_extractor.embed_image(image).float()
                # prediction 정의
                clip_voxels, clip_voxels_proj = diffusion_prior.voxel2clip(fmri_vol)
                clip_voxels = clip_voxels.view(len(fmri_vol),-1,clip_size) # [B, (257 * 768)] -> [B, 257, 768]

                #### forward 계산 + loss 계산 ####
                # forward(MLP backbone + Diffusion prior) -> prior loss
                prior_loss, clip_voxels_prediction = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target) # text(prediction) = fMRI, image(label) = image
                clip_voxels_prediction = clip_voxels_prediction / diffusion_prior.image_embed_scale # scale한 것을 원상태로 대돌려 놓음
                
                # forward(MLP projector) -> contrstive loss(mixco_nce_loss + soft_clip_loss)
                clip_voxels_norm = nn.functional.normalize(clip_voxels_proj.flatten(1), dim=-1) # cosine simility를 위해 미리 nomalization
                clip_target_norm = nn.functional.normalize(clip_target.flatten(1), dim=-1) # cosine simility를 위해 미리 nomalization
                # mixco_nce_loss(1/3) + soft_loss_temps(2/3)
                if epoch < int(mixup_pct * num_epochs):
                    nce_loss = mixco_nce_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=.006, 
                        perm=perm, betas=betas, select=select)
                else:
                    soft_loss_temps = cosine_anneal(0.004, 0.0075, num_epochs - int(mixup_pct * num_epochs))
                    epoch_temp = soft_loss_temps[epoch-int(mixup_pct*num_epochs)]
                    nce_loss = soft_clip_loss(
                        clip_voxels_norm,
                        clip_target_norm,
                        temp=epoch_temp)

                # 최종 loss 정의 
                loss = nce_loss + (prior_loss_coefficient * prior_loss)

                #### backward 계산 + update ####
                # gradient 계산 - amp사용
                scaler.scale(loss).backward() # amp사용

                # optimizer update - amp사용
                scaler.step(optimizer) # amp사용
                scaler.update() # amp사용
                torch.cuda.empty_cache() # gpu 메모리 cache삭제
                gc.collect() # # gpu 메모리 안 쓰는거 삭제

                # learning rate schedule update
                lr_scheduler.step()

                #### log ####
                # loss, lr 담아두기
                losses.append(loss.item())
                lrs.append(optimizer.param_groups[0]['lr'])

                # loss 누적 합
                loss_nce_sum += nce_loss.item() 
                loss_prior_sum += prior_loss.item()

                # cosinesimilarity 누적 합
                sims_base += nn.functional.cosine_similarity(clip_target_norm,clip_voxels_norm).mean().item() # item(): tensor에서 값만 출력 ex) torch.tensor(0.425, requires_grad=True)에서 0.425출력
                
                # top k 계산
                labels = torch.arange(len(clip_target_norm)).to(device) 
                fwd_percent_correct += topk(batchwise_cosine_similarity(clip_voxels_norm, clip_target_norm), labels, k=1)
                bwd_percent_correct += topk(batchwise_cosine_similarity(clip_target_norm, clip_voxels_norm), labels, k=1)

                logs = {
                    "train/num_steps": index + 1,
                    "train/lr": lrs[-1],
                    "train/loss": losses[-1],
                    "train/loss_nce": nce_loss,
                    "train/loss_prior": prior_loss,
                    "train/cosine_sim_mean": sims_base / len(losses),
                    "train/fwd_pct_correct_mean": fwd_percent_correct / len(losses),
                    "train/bwd_pct_correct_mean": bwd_percent_correct / len(losses),
                }
                progress_bar.set_postfix(**logs) # cli에 시각화
                wandb.log(logs) # wandb에 시각화

    return diffusion_prior

# def evaluate(args, data, models, saved_model_name):

#     device = args.device
#     num_epochs = args.num_epochs
#     clip_size = args.clip_size

#     scaler = GradScaler() # autocast scaler 인스턴스 생성
    
#     # model 정의 + train된 파라미터 불러오기
#     clip_extractor = models["clip"]

#     model_path = os.path.join(args.root_dir, args.output_dir, saved_model_name)
#     state_dict = torch.load(model_path, map_location=args.device) # 파라미터 불러오기
#     diffusion_prior = models["diffusion_prior"].load_state_dict(state_dict)
    
#     vd_pipe = models["vd_pipe"]
#     unet = models["unet"]
#     vae = models["vae"]
#     noise_scheduler = models["noise_scheduler"]

#     # log list
#     losses, lrs = [], []

#     # log list
#     losses, lrs = [], []

#     # 병목 추적
#     profiler = torch.profiler.profile(
#         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
#         on_trace_ready=torch.profiler.tensorboard_trace_handler("./logdir"),
#         record_shapes=True,
#         profile_memory=True,
#         with_stack=True,
#         use_cuda=True,
#     )
#     profiler.start() # 병목 조사 시작

#     progress_bar = tqdm(range(0,num_epochs), ncols=1200)
#     for epoch in progress_bar:
#         diffusion_prior.eval()

#         for index, (fmri_vol, image) in enumerate(data): # enumerate: index와 값을 같이 반환
#             with torch.no_grad():
#                 # data gpu 올리기
#                 fmri_vol = fmri_vol.to(device)   # fMRI -> GPU
#                 image = image.to(device)         # Image -> GPU

#                 # target 정의
#                 clip_target = clip_extractor.embed_image(image).float()
#                 # prediction 정의
#                 clip_voxels, clip_voxels_proj = diffusion_prior.voxel2clip(fmri_vol)
#                 clip_voxels = clip_voxels.view(len(fmri_vol),-1,clip_size) # [B, (257 * 768)] -> [B, 257, 768]

                
