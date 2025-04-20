import os
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn

from utils import img_augment, mixup, mixco_nce_loss, cosine_anneal, soft_clip_loss, topk, batchwise_cosine_similarity

def train(args, data, models, optimizer, lr_scheduler):
    
    device = args.device
    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    clip_size = args.clip_size
    prior_loss_coefficient = args.prior_loss_coefficient
    
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
            
            # image augmentation
            image = img_augment(image)

            # epoch의 1/3 지점 까지만 mixup 사용
            if epoch < int(mixup_pct * num_epochs):
                fmri_vol, perm, betas, select = mixup(fmri_vol)
            
            # target 정의
            clip_target = clip_extractor.embed_image(image).float()
            # prediction 정의
            clip_voxels, clip_voxels_proj = diffusion_prior.voxel2clip(fmri_vol)
            clip_voxels = clip_voxels.view(len(fmri_vol),-1,clip_size) # [B, (257 * 768)] -> [B, 257, 768]

            #### forward 계산 + loss 계산 ####
            # forward(MLP backbone + Diffusion prior) -> prior loss
            prior_loss, clip_voxels_prediction = diffusion_prior(text_embed=clip_voxels, image_embed=clip_target) # text(prediction) = fMRI, image(label) = image
            clip_voxels_prediction /= diffusion_prior.image_embed_scale # scale한 것을 원상태로 대돌려 놓음
            
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
            # gradient 계산 
            loss.backword()
            # optimizer update
            optimizer.step()
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
            progress_bar.set_postfix(**logs)
