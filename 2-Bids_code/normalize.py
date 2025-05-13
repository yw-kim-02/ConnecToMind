import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

for subj_num in range(1, 9):  # sub-01 ~ sub-08
    subj = f"sub-{subj_num:02d}"
    input_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/b4_roi/{subj}'  # 원본 폴더

    print(f"\nProcessing {subj}...")

    for sess in tqdm(range(1, 41), desc=f"{subj} sessions"):
        fname = f'betas_session{sess:02d}.nii.gz'
        input_path = os.path.join(input_dir, fname)  # 원본 파일 경로

        # output 경로 구성: sub-01/ses-01/func/
        sess_id = f"ses-{sess:02d}"
        output_dir = os.path.join(
            f'/nas/research/03-Neural_decoding/3-bids/derivatives/new_b4_roi_zscore/{subj}/{sess_id}/func'
        )
        os.makedirs(output_dir, exist_ok=True)

        # 저장 파일 이름 예: sub-01_ses-01_desc-betaroizscore.nii.gz
        output_fname = f"{subj}_{sess_id}_desc-betaroizscore.nii.gz"
        output_path = os.path.join(output_dir, output_fname)

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        if os.path.exists(output_path):
            continue  # 이미 존재하면 스킵

        img = nib.load(input_path)
        data = img.get_fdata()

        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std[std == 0] = 1e-8
        zscore_data = (data - mean) / std

        zscore_img = nib.Nifti1Image(zscore_data, affine=img.affine, header=img.header)
        nib.save(zscore_img, output_path)

print("\n모든 subject 정규화 완료!")
