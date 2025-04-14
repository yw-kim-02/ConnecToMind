

import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import shutil

for subj_num in range(8, 9):  # sub-01 ~ sub-08
    subj = f"sub-{subj_num:02d}"
    input_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/normalize/{subj}'    #원본 폴더 경로 
    output_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/b4_roi_zscore/{subj}/func'    #정규화 적용 저장 폴더 경로 
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {subj}...")

    for sess in tqdm(range(1, 41), desc=f"{subj} sessions"):  # sub별로 40 sessions 반복 
        fname = f'betas_session{sess:02d}.nii.gz'
        input_path = os.path.join(input_dir, fname) #원본 경로 
        output_path = os.path.join(output_dir, f'zscore_{fname}')   #결과 저장 경로

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        if os.path.exists(output_path):
            continue  # 이미 정규화된 파일은 스킵

        shutil.copyfile(input_path, output_path)    #원본 파일 출력 디렉토리로 복사
        img = nib.load(output_path)   #복사한 파일을 불러옴 
        data = img.get_fdata()  #위 데이터를 numpy array로 변환

        mean = np.mean(data, axis=-1, keepdims=True)    #run 기준으로 평균 계산
        std = np.std(data, axis=-1, keepdims=True)  #run 기준으로 표준편차 계산
        std[std == 0] = 1e-8  #표준편차가 0인 경우를 방지하기 위해 작은 값으로 대체

        zscore_data = (data - mean) / std   #zscore 정규화
        zscore_img = nib.Nifti1Image(zscore_data, affine=img.affine, header=img.header) #정규화된 데이터를 새로운 nii 파일로 생성
        nib.save(zscore_img, output_path)   #정규화된 nii 파일을 output path에 저장


print("\n모든 subject 정규화 완료!")

'''
import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import shutil
from multiprocessing import Pool

def process_subject(subj_num):
    subj = f"sub-{subj_num:02d}"
    input_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/normalize/{subj}'
    output_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/b4_roi_zscore/{subj}/func'
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nProcessing {subj}...")

    for sess in range(1, 41):
        fname = f'betas_session{sess:02d}.nii.gz'
        input_path = os.path.join(input_dir, fname)
        output_path = os.path.join(output_dir, f'zscore_{fname}')

        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            continue

        if os.path.exists(output_path):
            continue

        shutil.copyfile(input_path, output_path)
        img = nib.load(output_path)
        data = img.get_fdata()

        mean = np.mean(data, axis=-1, keepdims=True)
        std = np.std(data, axis=-1, keepdims=True)
        std[std == 0] = 1e-8

        zscore_data = (data - mean) / std
        zscore_img = nib.Nifti1Image(zscore_data, affine=img.affine, header=img.header)
        nib.save(zscore_img, output_path)

    print(f"{subj} 완료.")

if __name__ == '__main__':
    from multiprocessing import set_start_method
    set_start_method("spawn", force=True)  # Mac/Linux 호환성 위해 필요할 수 있음

    subj_nums = list(range(2, 9))  # sub-02 ~ sub-08 (총 7개)
    with Pool(processes=7) as pool:
        pool.map(process_subject, subj_nums)

    print("\n모든 subject 정규화 완료!")
'''
