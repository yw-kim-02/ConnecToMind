import os
import sys
import json
from args import parse_args
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

import utils
from data import stack_sub1_dataset
from models import Clipper, BrainNetwork, BrainDiffusionPrior, VersatileDiffusionPriorNetwork
from optimizers import get_optimizer
from schedulers import get_scheduler

def train():
    args = parse_args()

    if args.subj == 1:
        num_voxels = 15724
    elif args.subj == 2:
        num_voxels = 14278
    elif args.subj == 3:
        num_voxels = 15226
    elif args.subj == 4:
        num_voxels = 13153
    elif args.subj == 5:
        num_voxels = 13039
    elif args.subj == 6:
        num_voxels = 17907
    elif args.subj == 7:
        num_voxels = 12682
    elif args.subj == 8:
        num_voxels = 14386

    clip_size = 768
    
    # clip 정의
    clip_extractor = Clipper(lip_variant=args.clip_variant, norm_embs=args.norm_embs, hidden_state=args.hidden_state, device=args.device)
    
    # brain network 정의의
    out_dim = 257 * clip_size
    voxel2clip_kwargs = dict(in_dim=num_voxels, out_dim=out_dim, clip_size=clip_size)
    voxel2clip = BrainNetwork(**voxel2clip_kwargs)

    # difussion prior 정의
    out_dim = clip_size # 모델 거치면 257 * 768 나옴
    depth = 6 # transformer의 layer 수 -> versatile의 한 step의 layer
    dim_head = 64 # head당 dim
    heads = clip_size//dim_head # attention head 수
    timesteps = 100 # difusion step 수 
    guidance_scale = 3.5 # cfg inference할 때 사용 
    
    prior_network = VersatileDiffusionPriorNetwork(
            dim=out_dim,
            depth=depth,
            dim_head=dim_head,
            heads=heads,
            causal=False,
            num_tokens = 257,
            learned_query_mode="pos_emb"
        ).to(args.device)
    print("prior_network loaded")

    diffusion_prior = BrainDiffusionPrior(
        net=prior_network,
        image_embed_dim=out_dim,
        condition_on_text_encodings=False,
        timesteps=timesteps,
        cond_drop_prob=0.2,
        image_embed_scale=None,
        voxel2clip=voxel2clip,
    ).to(args.device)

    # optimizer
    optimizer = get_optimizer(diffusion_prior, lr=args.max_lr, optimizer_name=args.optimizer)

    # scheduler(train만 사용)
    train_dataset, _ = stack_sub1_dataset()
    num_train = len(train_dataset)
    lr_scheduler = get_scheduler(
        optimizer=optimizer,
        scheduler_type=args.lr_scheduler_type,
        num_epochs=args.num_epochs,
        num_train=num_train,
        batch_size=args.batch_size,
        num_devices=args.num_devices,
        max_lr=args.max_lr
    )