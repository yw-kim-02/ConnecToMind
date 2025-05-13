import os
import numpy as np
import nibabel as nib
from tqdm import tqdm

subj = 'sub-01'
mask_path = f'/nas/research/03-Neural_decoding/3-bids/derivatives/new_b4_roi_zscore/{subj}/sub-01_nsdgeneral.nii.gz'
mask = nib.load(mask_path).get_fdata()
mask_bool = mask > 0  # True for ROI voxel

base_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/new_b4_roi_zscore/{subj}'

for sess_num in range(1, 41):
    sess_id = f"ses-{sess_num:02d}"
    nii_path = os.path.join(base_dir, sess_id, "func", f"{subj}_{sess_id}_desc-betaroizscore.nii.gz")
    if not os.path.exists(nii_path):
        print(f"파일 없음: {nii_path}")
        continue

    img = nib.load(nii_path)
    data = img.get_fdata()  # shape: (X, Y, Z, T)
    
    # 마스크 적용 → shape: (N_voxels, T)
    masked_data = data[mask_bool, :]

    # Transpose → shape: (T, N_voxels)
    masked_data_T = masked_data.T.astype(np.float32)

    # 저장 경로 설정
    npy_path = os.path.join(base_dir, sess_id, "func", f"{subj}_{sess_id}_desc-betaroizscore.npy")
    np.save(npy_path, masked_data_T)
    print(f"저장 완료: {npy_path}  shape: {masked_data_T.shape}")