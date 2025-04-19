import os
from args import parse_args
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm


def train(models, data):
    args = parse_args()


    num_epochs = args.num_epochs
    mixup_pct = args.mixup_pct
    clip_size = args.clip_size
    lr_scheduler_type = args.lr_scheduler_type
    
    clip_extractor = models["clip"]
    diffusion_prior = models["diffusion_prior"]
    optimizer = models["optimizer"]
    lr_scheduler = models["scheduler"]
    
    progress_bar = tqdm(range(0,num_epochs), ncols=1200)
    for epoch in progress_bar:
        diffusion_prior.train()

        # metric
        sims_base = 0. # cosinesimilarity[(fMRI → CLIP), (image → CLIP)]의 누적합 -> 평균 구할 때 쓰임
        fwd_percent_correct = 0. # forward prediction이 정답과의 cs가가 가장 높으면 1, 아니면 0 -> 비율의 누적합 -> 평균 구할 때 쓰임
        bwd_percent_correct = 0. # backward prediction이 정답과의 cs가가 가장 높으면 1, 아니면 0 -> 비율의 누적합 -> 평균 구할 때 쓰임
        loss_nce_sum = 0. # Negative Contrastive Estimation loss의 누적합 -> 평균 구할 때 쓰임
        loss_prior_sum = 0. # MSE loss의 누적합 -> 평균 구할 때 쓰임

        for index, (fmri_vol, image) in enumerate(data):

    return