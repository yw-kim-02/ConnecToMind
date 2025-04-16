import os
import re
import glob
import pandas as pd
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import ConcatDataset
from torch import nn
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import nibabel as nib

import utils
from args import parse_args


class FmriImageDataset(Dataset):
    def __init__(self, fmri_path, tsv_path, image_path, train=1, transform=None):
        self.fmri_path = fmri_path
        self.tsv_path = tsv_path
        self.image_path = image_path
        self.train = train  # 'train' or 'test'
        self.transform = transform

        # train & test 각각 index 뽑아두기
        df = pd.read_csv(self.tsv_path, sep='\t')
        self.valid_indices = df[df['train'] == train].index.tolist()

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        actual_idx = self.valid_indices[idx]  # 전체 데이터에서의 진짜 인덱스

        # 해당 볼륨만 로드 (4D → 3D)
        fmri = nib.load(self.fmri_path).get_fdata()
        fmri_vol = torch.tensor(fmri[:, :, :, actual_idx]).float()

        # image column 한 행만 로딩
        row = pd.read_csv(self.tsv_path, sep='\t', skiprows=range(1, actual_idx + 1), nrows=1)
        image_id = row['image'].values[0]

        # 이미지 로딩
        image_path = os.path.join(self.image_path, image_id + '.jpg')
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return fmri_vol, image

def stack_sub1_dataset():
    args = parse_args()

    root_dir = args.root_dir
    fmri_dir = args.fmri_dir
    fmri_detail_dir = args.fmri_detail_dir
    image_dir = args.image_dir
    
    # 세션 자동 추출
    pattern = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-*/func/sub-01_ses-*_desc-betaroizscore.nii.gz"
    fmri_files = glob.glob(pattern)
    sessions = sorted([re.search(r'ses-(\d+)', f).group(1) for f in fmri_files]) # ex) 01,02 ... 추출

    train_datasets, test_datasets = [], []

    for ses in sessions:
        fmri_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-{ses}/func/sub-01_ses-{ses}_desc-betaroizscore.nii.gz"
        tsv_path = f"{root_dir}/{fmri_dir}/{fmri_detail_dir}/sub-01/ses-{ses}/func/sub-01_ses-{ses}_task-image_events.tsv"
        image_path = f"{root_dir}/{image_dir}"
        
        train_datasets.append(FmriImageDataset(fmri_path, tsv_path, image_path, train=1))
        test_datasets.append(FmriImageDataset(fmri_path, tsv_path, image_path, train=0))

    # Dataset 합침
    train_dataset = ConcatDataset(train_datasets)
    test_dataset = ConcatDataset(test_datasets)
    
    return train_dataset, test_dataset


def get_loader():
    args = parse_args()

    # 시드 고정
    utils.seed_everything(args.seed) 
    
    # Dataset 생성
    train_dataset, test_dataset = stack_sub1_dataset()

    # DataLoader 생성
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=args.is_shuffle)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=False)

    return train_loader, test_loader