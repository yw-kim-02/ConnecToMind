import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

from PIL import Image
from torch import nn
import torch.nn.functional as F


class Bids(Dataset):
    def __init__(self, args):
        self.args = args
        self.root_dir = args.root_dir

        # fMRI & 이벤트 파일을 한 번만 검색하여 저장
        self.fmri_paths = self._load_fmri_paths() 
        self.event_paths = self._load_event_paths()
        self.common_keys = sorted(set(self.fmri_paths.keys()) & set(self.event_paths.keys())) # 공통 키(태스크 이름) 찾기

        self.samples = self._load_samples()

        self.length = len(self.common_keys)

    def _load_fmri_paths(self):
        fmri_files = glob.glob(os.path.join(self.root_dir, "**", "*_task-image_run-*_bold.nii.gz"), recursive=True)
        return {os.path.basename(f).replace("_bold.nii.gz", ""): f for f in fmri_files}


    def _load_event_paths(self):
        event_files = glob.glob(os.path.join(self.root_dir, "**", "*_task-image_run-*_events.tsv"), recursive=True)
        return {os.path.basename(e).replace("_events.tsv", ""): e for e in event_files}

    def _load_image_paths(self, image_filenames):
        return [
            os.path.join(self.root_dir, "4-images", f"{img}.jpg")
            for img in image_filenames if os.path.exists(os.path.join(self.root_dir, "4-images", f"{img}.jpg"))
        ]

    def _load_samples(self):
        """
        {
            "sub-01_ses-01_task-image": {
                "fmri": "/path/bold.nii.gz",
                "events": "/path/events.tsv",
                "images": "/path/coco2014_000000488558"
            },
            ...
        }
        """
        samples = {}

        for key in self.common_keys:
            event_paths = self.event_paths[key]

            image_filenames = pd.read_csv(event_paths, sep="\t")["image"].dropna().astype(str).tolist()
            image_paths = self._load_image_paths(image_filenames)
        
            samples[key] = {
                "fmri": self.fmri_paths[key],
                "events": self.event_paths[key],
                "images": image_paths
            }

        return samples