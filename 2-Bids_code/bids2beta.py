import os
import shutil  # 파일 복사용
import nibabel as nib
import numpy as np
import pandas as pd
import json
import multiprocessing
from tqdm import tqdm  # 진행률 표시
import warnings

from nilearn.glm.first_level import FirstLevelModel, make_first_level_design_matrix

warnings.filterwarnings("ignore")

# 원본 데이터 및 결과 저장 경로 설정
# bids_root = "/nas/research/03-Neural_decoding/3-bids/raw_data"  # BIDS 원본 데이터 경로
# derivatives_root = "/nas/research/03-Neural_decoding/3-bids/derivatives/raw_beta"
bids_root = "/nas/3-bids/raw_data"  # BIDS 원본 데이터 경로
derivatives_root = "/nas/3-bids/derivatives/raw_beta"

# BIDS 구조에서 subject 탐색
subjects = sorted([sub for sub in os.listdir(bids_root) if sub.startswith("sub-")])

def process_subject(sub):
    sub_path = os.path.join(bids_root, sub)
    sessions = [ses for ses in os.listdir(sub_path) if ses.startswith("ses-")]

    for ses in sessions:
        func_path = os.path.join(sub_path, ses, "func")

        if not os.path.exists(func_path):
            continue  # func 폴더 없으면 스킵

        # task bold.nii.gz 및 events.tsv 파일 검색
        func_files = [f for f in os.listdir(func_path) if f.endswith("_bold.nii.gz") and "task-image" in f]

        for bold_file in func_files:
            run_id = bold_file.split("_")[3]  # run-01, run-02 등 추출
            events_file = bold_file.replace("_bold.nii.gz", "_events.tsv")
            json_file = bold_file.replace("_bold.nii.gz", "_bold.json")  # JSON 파일에서 TR 읽기

            events_path = os.path.join(func_path, events_file)
            json_path = os.path.join(func_path, json_file)
            
            # 저장할 beta 파일 경로 설정
            beta_save_path = os.path.join(derivatives_root, sub, ses, "func")
            beta_file = os.path.join(beta_save_path, f"{sub}_{ses}_task-image_{run_id}_desc-beta.nii.gz")

            # ✅ 이미 처리된 데이터가 있으면 건너뛰기
            if os.path.exists(beta_file):
                print(f"[{sub}] Skipping: {beta_file} already exists.")
                continue
            
            if not os.path.exists(events_path) or not os.path.exists(json_path):
                continue  # 이벤트 파일 또는 JSON 파일이 없으면 스킵
            
            # fMRI 데이터 로드
            bold_path = os.path.join(func_path, bold_file)
            fmri_img = nib.load(bold_path)

            # TR 값 JSON에서 읽기
            with open(json_path, "r") as f:
                metadata = json.load(f)
            tr = metadata.get("RepetitionTime")
            if isinstance(tr, list):
                tr = tr[0]

            #  이벤트 파일 로드
            events = pd.read_csv(events_path, sep='\t')
            events = events.copy()
            events['trial_type'] = ['trial_' + str(i) for i in range(len(events))]
            
            # 디자인 매트릭스 생성 (Nilearn의 기본 사용)
            n_scans = fmri_img.shape[-1]
            frame_times = np.arange(n_scans) * tr
            design_matrix = make_first_level_design_matrix(frame_times, events, drift_model=None, high_pass=None)

            trial_columns = [col for col in design_matrix.columns if "trial_" in col]
            design_matrix = design_matrix[trial_columns]
            
            # FirstLevelModel 생성
            first_level_model = FirstLevelModel(t_r=tr, drift_model=None, high_pass=None)
            
            # 모델 피팅
            first_level_model = first_level_model.fit(fmri_img, events=events, design_matrices=design_matrix)
            
            # 각 trial에 대해 contrast 계산하여 단일-트라이얼 베타 맵 생성
            trial_beta_imgs = []
            for i, col in enumerate(design_matrix.columns):
                contrast_vec = np.zeros(len(trial_columns))
                contrast_vec[trial_columns.index(col)] = 1
                contrast_img = first_level_model.compute_contrast(contrast_vec, output_type='effect_size')
                trial_beta_imgs.append(contrast_img)
            
            # 3D 이미지들을 4D 이미지로 결합
            beta_4d = nib.concat_images(trial_beta_imgs)

            # 개수 맞는지 확인
            if beta_4d.shape[-1] != len(events):
                print(f"[경고] {sub}, {ses}: Beta volume 개수 불일치! ({beta_4d.shape[-1]} vs {len(events)})")
                continue
            
            # 저장 경로 생성
            os.makedirs(beta_save_path, exist_ok=True)
            nib.save(beta_4d, beta_file)
            
            # events.tsv 파일도 derivatives 폴더에 저장
            events_save_path = os.path.join(beta_save_path, f"{sub}_{ses}_task-image_{run_id}_events.tsv")
            shutil.copy(events_path, events_save_path)
            
            print(f"[{sub}] Saved: {beta_file} with shape {beta_4d.shape} (TR = {tr}s)")
            print(f"[{sub}] Saved: {events_save_path} (events.tsv copied)")

if __name__ == "__main__":
    # ✅ 처리되지 않은 subject만 필터링
    subjects_to_process = []
    for sub in subjects:
        sub_path = os.path.join(bids_root, sub)
        sessions = [ses for ses in os.listdir(sub_path) if ses.startswith("ses-")]

        for ses in sessions:
            beta_save_path = os.path.join(derivatives_root, sub, ses, "func")
            func_files = [f for f in os.listdir(os.path.join(sub_path, ses, "func")) if f.endswith("_bold.nii.gz") and "task-image" in f]

            for bold_file in func_files:
                run_id = bold_file.split("_")[3]
                beta_file = os.path.join(beta_save_path, f"{sub}_{ses}_task-image_{run_id}_desc-beta.nii.gz")

                if not os.path.exists(beta_file):  # ✅ 처리되지 않은 경우만 추가
                    subjects_to_process.append(sub)
                    break  # 한 개라도 미처리된 데이터가 있으면 추가하고 넘어감


    # ✅ 건너뛴 데이터 제외하고 진행률 표시
    with multiprocessing.Pool(processes=20) as pool:
        for _ in tqdm(pool.imap(process_subject, subjects_to_process), total=len(subjects_to_process), desc="Processing Subjects"):
            pass