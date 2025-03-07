import os
import re
from tqdm import tqdm
import subprocess

# 원본 데이터와 출력 폴더 경로
source_dir = "/nas-tmp/research/03-Neural_decoding/1-raw_data/nsd"
output_dir = "/nas-tmp/research/03-Neural_decoding/3-bids"

# 피험자 목록 (예: 8명의 피험자)
subjects = [d for d in os.listdir(source_dir) if d.startswith("sub-")] # sub모음
os.makedirs(output_dir, exist_ok=True)

for sub in tqdm(subjects, desc="Processing subjects"):
    sub_dir = os.path.join(output_dir, sub)  # BIDS 폴더: sub
    os.makedirs(sub_dir, exist_ok=True)

    subject_path = os.path.join(source_dir, sub)  # Raw 폴더: sub
    sessions = [d for d in os.listdir(subject_path) if d.startswith("ses-") and d[-2:].isdigit()] # ses모음

    # sessions 처리
    for i, ses in tqdm(enumerate(sessions,start=1), desc=f"Processing {sub}"):
        ses_dir = os.path.join(sub_dir, f"ses-{i:02d}")  # BIDS 폴더: sub-ses
        os.makedirs(ses_dir, exist_ok=True)
        
        session_path = os.path.join(subject_path, ses)  # Raw 폴더: sub-ses
        
        # anat 폴더 처리
        anat_dir = os.path.join(ses_dir, "anat")
        os.makedirs(anat_dir, exist_ok=True)
        source_anat_nii_path = os.path.join(subject_path, "ses-nsdanat/anat", f"{sub}_ses-nsdanat_run-01_T1w.nii.gz") # Raw 폴더: sub-ses-anat
        source_anat_json_path = os.path.join(subject_path, "ses-nsdanat/anat", f"{sub}_ses-nsdanat_run-01_T1w.json") # Raw 폴더: sub-ses-anat
        # T1w.nii.gz 파일 복사
        target_anat_nii_filename = f"{sub}_ses-{i:02d}_T1w.nii.gz"
        target_anat_nii_path = os.path.join(anat_dir, target_anat_nii_filename)
        if os.path.exists(source_anat_nii_path) and not os.path.exists(target_anat_nii_path):
            subprocess.run(["rsync", "-avz", source_anat_nii_path, target_anat_nii_path])
        # T1w.json 파일 복사
        target_anat_json_filename = f"{sub}_ses-{i:02d}_T1w.json"
        target_anat_json_path = os.path.join(anat_dir, target_anat_json_filename)
        if os.path.exists(source_anat_json_path) and not os.path.exists(target_anat_json_path):
            subprocess.run(["rsync", "-avz", source_anat_json_path, target_anat_json_path])
            
        # dwi 폴더 처리
        dwi_dir = os.path.join(ses_dir, "dwi")
        os.makedirs(dwi_dir, exist_ok=True)
        
        dwi_files = [
            f"{sub}_ses-nsddiffusion_acq-98_dir-AP_dwi.bval",
            f"{sub}_ses-nsddiffusion_acq-98_dir-AP_dwi.bvec",
            f"{sub}_ses-nsddiffusion_acq-98_dir-AP_dwi.json",
            f"{sub}_ses-nsddiffusion_acq-98_dir-AP_dwi.nii.gz",
            f"{sub}_ses-nsddiffusion_acq-98_dir-PA_dwi.bval",
            f"{sub}_ses-nsddiffusion_acq-98_dir-PA_dwi.bvec",
            f"{sub}_ses-nsddiffusion_acq-98_dir-PA_dwi.json",
            f"{sub}_ses-nsddiffusion_acq-98_dir-PA_dwi.nii.gz",
            f"{sub}_ses-nsddiffusion_acq-99_dir-AP_dwi.bval",
            f"{sub}_ses-nsddiffusion_acq-99_dir-AP_dwi.bvec",
            f"{sub}_ses-nsddiffusion_acq-99_dir-AP_dwi.json",
            f"{sub}_ses-nsddiffusion_acq-99_dir-AP_dwi.nii.gz",
            f"{sub}_ses-nsddiffusion_acq-99_dir-PA_dwi.bval",
            f"{sub}_ses-nsddiffusion_acq-99_dir-PA_dwi.bvec",
            f"{sub}_ses-nsddiffusion_acq-99_dir-PA_dwi.json",
            f"{sub}_ses-nsddiffusion_acq-99_dir-PA_dwi.nii.gz"
        ]
        
        for dwi_file in tqdm(dwi_files, desc=f"Processing {sub} ses-{i:02d}"):
            source_dwi_path = os.path.join(subject_path, "ses-nsddiffusion/dwi", dwi_file) # Raw 폴더: sub-sesnsddiffusion-dwi
            target_dwi_path = os.path.join(dwi_dir, dwi_file.replace("ses-nsddiffusion", f"ses-{i:02d}")) # BIDS 폴더: sub-ses-dwi
            if os.path.exists(source_dwi_path) and not os.path.exists(target_dwi_path):
                subprocess.run(["rsync", "-avz", source_dwi_path, target_dwi_path])
                
        # func 폴더 처리
        func_dir = os.path.join(ses_dir, "func") # BIDS 폴더: sub-ses-func
        os.makedirs(func_dir, exist_ok=True)
        func_path = os.path.join(session_path, "func") # Raw 폴더: sub-ses-func
        if os.path.exists(func_path):
            for func_file in tqdm(os.listdir(func_path), desc=f"Processing {sub} ses-{i:02d} func"):
                func_source = os.path.join(func_path, func_file)

                # run 숫자 두 자리로 변환 (run-1 → run-01)
                func_file = re.sub(r'run-(\d+)', lambda m: f"run-{int(m.group(1)):02d}", func_file)

                func_target = os.path.join(func_dir, func_file.replace(f"ses-nsd{i:02d}", f"ses-{i:02d}").replace("task-nsdcore", "task-image").replace("task-rest", "rest"))
                if not os.path.exists(func_target):
                    subprocess.run(["rsync", "-avz", func_source, func_target])