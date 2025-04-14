import os
import nibabel as nib
import numpy as np

for sub_num in range(1, 9):  # sub-03 ~ sub-08
    sub_id = f"sub-{sub_num:02d}"
    src_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/b4_roi/{sub_id}'    #원본 경로
    masked_dir = f'/nas/research/03-Neural_decoding/3-bids/derivatives/normalize/{sub_id}'   #마스크 적용 결과 저장 경로
    os.makedirs(masked_dir, exist_ok=True)

    mask_img = nib.load(os.path.join(src_dir, 'nsdgeneral.nii.gz')) #nsdgeneral.nii.gz 파일을 불러옴
    mask_data = mask_img.get_fdata()    #위 데이터를 numpy array로 변환
    roi_mask = (mask_data == 1) #값이 1인 위치는 true, 나머지는 false

    #마스킹 함수
    def apply_mask(input_path, output_path):
        img = nib.load(input_path)  #input path에 있는 파일을 불러옴
        data = img.get_fdata()  #위 데이터를 numpy array로 변환
        masked = np.where(roi_mask[..., np.newaxis], data, 0)   #3d -> 4d, true ? data : 0
        masked_img = nib.Nifti1Image(masked, affine=img.affine, header=img.header)  #마스킹된 데이터를 새로운 nii 파일로 생성
        nib.save(masked_img, output_path)   #마스킹된 nii 파일을 output path에 저장

    print(f"\nProcessing {sub_id}...")

    for fname in os.listdir(src_dir):
        if fname.startswith('betas_session') and fname.endswith('.nii.gz'):
            input_path = os.path.join(src_dir, fname)   #input 파일의 경로 설정
            output_path = os.path.join(masked_dir, fname)   #output 파일의 경로 설정

            # 이미 처리된 파일 건너뛰기 
            if os.path.exists(output_path):
                print(f"  {fname} already exists")
                continue

            print(f"  Masking {fname}...")
            apply_mask(input_path, output_path) #마스크 적용

print("모든 subject의 마스킹 완료!")