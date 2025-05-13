import os
import pandas as pd
from glob import glob

# shared.txt 읽기
shared_txt_path = '/nas/research/03-Neural_decoding/4-image/raw/nsd/info/shared.txt'
with open(shared_txt_path, 'r') as f:
    shared_images = {line.strip().replace('.jpg', '') for line in f.readlines()}

# 경로 설정
raw_base = '/nas/research/03-Neural_decoding/3-bids/raw_data'
save_base = '/nas/research/03-Neural_decoding/3-bids/derivatives/b4_roi_zscore_bids'

subs = range(1, 9)  # sub-01 ~ sub-08

for sub in subs:
    subj_id = f"sub-{sub:02d}"
    subj_dir = os.path.join(raw_base, subj_id)

    for ses in os.listdir(subj_dir):
        if not ses.startswith("ses-"):
            continue
        
        func_path = os.path.join(subj_dir, ses, "func")
        if not os.path.exists(func_path):
            continue

        # 저장 경로 및 파일명 구성
        save_dir = os.path.join(save_base, subj_id, ses, "func")
        os.makedirs(save_dir, exist_ok=True)

        output_filename = f"{subj_id}_{ses}_task-image_events.tsv"
        output_path = os.path.join(save_dir, output_filename)

        # 이미 존재하는 경우 스킵
        if os.path.exists(output_path):
            print(f"Already exists, skipping: {output_path}")
            continue

        # func 내 *_events.tsv 찾기
        tsv_files = glob(os.path.join(func_path, '*_events.tsv'))
        image_rows = []

        for tsv_file in tsv_files:
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                if 'image' in df.columns:
                    image_rows.append(df[['image']])
            except Exception as e:
                print(f"Error reading {tsv_file}: {e}")
        
        if image_rows:
            merged_df = pd.concat(image_rows, ignore_index=True)

            # 'train' 컬럼 추가
            merged_df['train'] = merged_df['image'].apply(lambda x: 0 if x in shared_images else 1)

            # 저장
            merged_df.to_csv(output_path, sep='\t', index=False)
            print(f"Saved: {output_path}")
