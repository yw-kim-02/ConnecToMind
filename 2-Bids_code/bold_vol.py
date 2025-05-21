import os
import numpy as np
import nibabel as nib
import pandas as pd
from tqdm import tqdm

# 입력 및 출력 디렉토리
input_root = '/nas/research/03-Neural_decoding/3-bids/derivatives/raw_prep'  # 기존 NIfTI 데이터가 저장된 루트
event_root = '/nas/research/03-Neural_decoding/3-bids/raw_data'  # 이벤트 tsv 파일이 저장된 루트
output_root = '/nas/research/03-Neural_decoding/3-bids/derivatives/raw_prep_vol'  # 결과 저장 디렉토리

sub_id = "sub-01"

# 4의 배수 목록
expected_onsets = list(range(12, 284, 4))

for ses_num in tqdm(range(1, 41)):
    ses_id = f"ses-{ses_num:02d}"
    session_data_list = []

    for run_num in range(1, 13):
        run_id = f"run-{run_num:02d}"
        func_dir = os.path.join(input_root, sub_id, ses_id, "func")
        event_dir = os.path.join(event_root, sub_id, ses_id, "func")
        output_dir = os.path.join(output_root, sub_id, ses_id, "func")
        os.makedirs(output_dir, exist_ok=True)

        nii_file = os.path.join(func_dir, f"{sub_id}_{ses_id}_task-image_{run_id}_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz")
        event_file = os.path.join(event_dir, f"{sub_id}_{ses_id}_task-image_{run_id}_events.tsv")

        if not os.path.exists(nii_file) or not os.path.exists(event_file):
            print(f"파일 누락: {nii_file} 또는 {event_file}")
            continue

        img = nib.load(nii_file)
        data = img.get_fdata()
        affine = img.affine
        header = img.header

        #TR이 5의 배수인 인덱스 제거
        total_TR = data.shape[-1]
        tr_indices = list(range(total_TR))
        filtered_tr_indices = [i for i in tr_indices if (i + 1) % 5 != 0]

        #앞 6개, 뒤 9개 제거
        filtered_tr_indices = filtered_tr_indices[6:-9]

        #empty stimuli 제거
        df = pd.read_csv(event_file, sep='\t')
        present_onsets = sorted(df['onset'].astype(int).tolist())
        missing_onsets = sorted(set(expected_onsets) - set(present_onsets))

        for mo in missing_onsets:
            idx = expected_onsets.index(mo)
            remove_idx_1 = idx * 2 - 1
            remove_idx_2 = idx * 2
            
            for rm in [remove_idx_1, remove_idx_2]:
                if 0 <= rm < len(filtered_tr_indices):
                    filtered_tr_indices[rm] = -1  # 삭제 마킹

        filtered_tr_indices = [i for i in filtered_tr_indices if i != -1]

        new_data = data[..., filtered_tr_indices]
        session_data_list.append(new_data)

    if len(session_data_list) > 0:
        #모든 run 데이터를 시간 축으로 concat
        concat_data = np.concatenate(session_data_list, axis=-1)
        print(f"{ses_id} shape: {concat_data.shape}")

        if concat_data.shape[-1] % 2 != 0:
            raise ValueError("TR 개수가 2로 나누어지지 않음")

        reshaped = concat_data.reshape(*concat_data.shape[:-1], -1, 2)
        data_mean = reshaped.mean(axis=-1)

        '''
        new_total_TR = concat_data.shape[-1]
        labels = np.tile([1,2,3,4], new_total_TR//4)
        if len(labels) < new_total_TR:
            labels = np.concatenate([labels, np.arange(1, new_total_TR % 4 + 1)])
        selected_indices = np.where((labels == 2) | (labels == 3))[0]
        subsampled_data = concat_data[..., selected_indices]
        print(f"{ses_id} shape after label-based downsampling: {subsampled_data.shape}")
        '''

        print(f"{ses_id} shape after downsampling: {data_mean.shape}")

        # 저장
        new_img = nib.Nifti1Image(data_mean, affine=affine, header=header)
        output_path = os.path.join(output_dir, f"{sub_id}_{ses_id}_task-image_desc-boldprepvol.nii.gz")
        nib.save(new_img, output_path)