import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image
import nibabel as nib

import utils
from args import parse_args

from torch import nn
import torch.nn.functional as F

# 베타값(t=trial개수)을 기준으로 한 코드
class Bids(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_dir = args.root_dir

        # fMRI & 이벤트 파일을 한 번만 검색하여 저장
        self.fmri_paths = self._load_fmri_paths() 
        self.event_paths = self._load_event_paths()

        self.sample_key = sorted(set(self.fmri_paths.keys()) & set(self.event_paths.keys())) # 공통 키(태스크 이름) 찾기
        self.sample_paths = self._load_sample_paths()

        self.samples= self._load_samples()

        self.length = len(self.sample_key)

    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        sample = self.samples[idx]

        # fMRI 데이터 로드
        fmri_path, tr = sample["fmri"], sample["tr"]
        fmri_data = nib.load(fmri_path).get_fdata()[..., tr]  # 특정 TR의 볼륨만 로드

        # 이미지 데이터 로드
        image = Image.open(sample["image"]).convert("RGB")

        return {"fmri": fmri_data, "image": image}

    def _load_fmri_paths(self):
        '''
        {'sub-01_ses-01_task-image_run-01': '/path/to/sub-01_ses-01_task-image_run-01_bold.nii.gz' ...}
        '''
        fmri_files = glob.glob(os.path.join(self.root_dir, "**", "*_task-image_run-*_bold.nii.gz"), recursive=True)
        return {os.path.basename(f).replace("_bold.nii.gz", ""): f for f in fmri_files}

    def _load_event_paths(self):
        '''
        {'sub-01_ses-01_task-image_run-01': '/path/to/sub-01_ses-01_task-image_run-01_events.tsv' ...}
        '''
        event_files = glob.glob(os.path.join(self.root_dir, "**", "*_task-image_run-*_events.tsv"), recursive=True)
        return {os.path.basename(e).replace("_events.tsv", ""): e for e in event_files}

    def _load_image_paths(self, image_filenames):
        '''
        ['/path/to/coco2017_46003.jpg', '/path/to/coco2017_61883.jpg' ...]
        '''
        return [
            os.path.join(self.root_dir, "4-image", f"{img}.jpg") for img in image_filenames 
        ]

    def _load_sample_paths(self):
        """
        {
            "sub-01_ses-01_task-image_run-01": {
                "fmri": "/path/to/sub-01_ses-01_task-image_run-01_bold.nii.gz",
                "images": ['/path/to/coco2017_46003.jpg', '/path/to/coco2017_61883.jpg' ...]
            },
            ...
        }
        """
        sample_paths = {}

        for key in self.sample_key:
            event_paths = self.event_paths[key]

            image_filenames = pd.read_csv(event_paths, sep="\t").sort_values(by="onset") # fMRI의 trial 순서대로 task image정렬
            image_filenames = image_filenames["image"].dropna().astype(str).tolist()
            image_paths = self._load_image_paths(image_filenames)

            sample_paths[key] = {
                "fmri": self.fmri_paths[key],
                "images": image_paths
            }

        return sample_paths
    
    def _load_samples(self):
        """
        모든 fMRI TR별 볼륨과 이미지 매핑을 생성하여 리스트로 저장
        [
            {"fmri": "/path/to/fmri.nii.gz", "tr": 0, "image": "/path/to/coco2017_612.jpg"},
            {"fmri": "/path/to/fmri.nii.gz", "tr": 1, "image": "/path/to/coco2017_618.jpg"},
            ...
        ]
        """
        # `.npy` 파일이 존재하면 로드
        save_path = os.path.join(self.root_dir, "5-mindeye_code", "mindeye1", "samples.npy")
        if os.path.exists(save_path):
            print(f"기존 `samples.npy` 파일 로드 중...")
            return np.load(save_path, allow_pickle=True).tolist()  # ✅ 리스트로 변환

        print(f"`samples.npy` 파일 없음 → 새로 생성 중...")

        samples = []

        for key, sample in self.sample_paths.items():
            fmri_path = sample["fmri"]
            image_paths = sample["images"]  # 정렬된 이미지 리스트

            # fMRI 데이터의 TR 개수 확인
            T = nib.load(fmri_path).shape[-1]

            # 모든 trial을 리스트에 추가
            for t in range(T):
                samples.append({
                    "fmri": fmri_path,
                    "tr": t,
                    "image": image_paths[t]
                })
        
        # npy로 저장
        os.makedirs(os.path.dirname(save_path), exist_ok=True)  
        np.save(save_path, samples) 
        
        return samples
    
def get_loader():
    args = parse_args()

    # 시드 고정
    utils.seed_everything(args.seed) 

    dataset = Bids(args)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=args.is_shuffle)

    return dataloader