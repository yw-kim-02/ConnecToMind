import os
import re
from tqdm import tqdm
import subprocess

# 원본 데이터와 출력 폴더 경로
source_dir = "/nas-tmp/research/03-Neural_decoding/1-raw_data/things"
output_dir = "/nas-tmp/research/03-Neural_decoding/3-bids"

# 피험자 목록 (예: 8명의 피험자)
subjects = [d for d in os.listdir(source_dir) if d.startswith("sub-")] # sub모음
os.makedirs(output_dir, exist_ok=True)

for i, sub in tqdm(enumerate(subjects,start=1), desc=f"Processing subject"):
    new_sub = f"sub-{i+17:02d}" # BIDS의 이름은 sub-<index>로 해야함
    sub_dir = os.path.join(output_dir, new_sub)  # BIDS 폴더: sub
    os.makedirs(sub_dir, exist_ok=True)

    subject_path = os.path.join(source_dir, sub)  # Raw 폴더: sub
    sessions = [d for d in os.listdir(subject_path) if d.startswith("ses-things") and d[-2:].isdigit()] # Raw 폴더 ses모음

    # sessions 처리
    for i, ses in tqdm(enumerate(sessions,start=1), desc=f"Processing {new_sub}"):
        ses_dir = os.path.join(sub_dir, f"ses-{i:02d}")  # BIDS 폴더: sub-ses
        os.makedirs(ses_dir, exist_ok=True)
        
        session_path = os.path.join(subject_path, ses)  # Raw 폴더: sub-ses
        
        # anat 폴더 처리
        anat_dir = os.path.join(ses_dir, "anat")
        os.makedirs(anat_dir, exist_ok=True)
        source_anat_nii_path = os.path.join(subject_path, "ses-localizer1/anat", f"{sub}_ses-localizer1_acq-prescannormalized_rec-pydeface_T1w.nii.gz") # Raw 폴더: sub-ses-anat
        source_anat_json_path = os.path.join(subject_path, "ses-localizer1/anat", f"{sub}_ses-localizer1_acq-prescannormalized_rec-pydeface_T1w.json") # Raw 폴더: sub-ses-anat
        
        # T1w.nii.gz 파일 복사
        if os.path.exists(source_anat_nii_path):
            target_anat_nii_filename = f"{new_sub}_ses-{i:02d}_T1w.nii.gz"
            target_anat_nii_path = os.path.join(anat_dir, target_anat_nii_filename)
            subprocess.run(["rsync", "-avz", source_anat_nii_path, target_anat_nii_path])

        # T1w.nii.gz 파일 복사
        if os.path.exists(source_anat_nii_path):
            target_anat_json_filename = f"{new_sub}_ses-{i:02d}_T1w.nii.json"
            target_anat_json_path = os.path.join(anat_dir, target_anat_json_filename)
            subprocess.run(["rsync", "-avz", source_anat_nii_path, target_anat_json_path])

        
        # dwi 폴더 처리
        dwi_dir = os.path.join(ses_dir, "dwi")
        os.makedirs(dwi_dir, exist_ok=True)
                
        # func 폴더 처리
        func_dir = os.path.join(ses_dir, "func") # BIDS 폴더: sub-ses-func
        os.makedirs(func_dir, exist_ok=True)
        func_path = os.path.join(session_path, "func") # Raw 폴더: sub-ses-func
        if os.path.exists(func_path):
            for func_file in tqdm(os.listdir(func_path), desc=f"Processing {new_sub} ses-{i:02d} func"):
                if "events" in func_file or "bold" in func_file:
                    func_source = os.path.join(func_path, func_file)
                    func_target = os.path.join(func_dir, func_file.replace(sub, new_sub).replace(f"{ses}", f"ses-{i:02d}").replace("task-things", "task-image").replace("_acq-reversePE", ""))
                    subprocess.run(["rsync", "-avz", func_source, func_target])