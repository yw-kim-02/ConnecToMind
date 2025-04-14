'''
import os
import pandas as pd
from glob import glob

base_dir = '/nas/research/03-Neural_decoding/3-bids/raw_data'

subs = range(1, 9)  # 1~8

for sub in subs:
    for ses in os.listdir(f"{base_dir}/sub-{sub:02d}"):
        if not ses.startswith("ses-"):
            continue
        func_path = f"{base_dir}/sub-{sub:02d}/{ses}/func"
        if not os.path.exists(func_path):
            continue

        # 저장 파일명: sub-XX_ses-YY_task-image_events.tsv

        save_dir = '/nas/research/03-Neural_decoding/3-bids/derivatives/b4_roi_zscore/sub-{sub:02d}/func'
        os.makedirs(save_dir, exist_ok=True)
        output_filename = f"sub-{sub:02d}_{ses}_task-image_events.tsv"
        output_path = os.path.join(save_dir, output_filename)
        
        if os.path.exists(output_path):
            print(f"Already exists, skipping: {output_path}")
            continue  # 이미 처리된 경우 건너뜀
        
        tsv_files = glob(os.path.join(func_path, '*_events.tsv'))   #func_path 폴더 내 events.tsv로 끝나는 파일 리스트로 반환 
        image_rows = [] #image col들 이어붙이기 위한 list 
        
        for tsv_file in tsv_files:
            try:
                df = pd.read_csv(tsv_file, sep='\t')
                if 'image' in df.columns:
                    image_rows.append(df[['image']])    #image col 있으면 리스트에 추가가
            except Exception as e:
                print(f"Error reading {tsv_file}: {e}")
        
        if image_rows:
            merged_df = pd.concat(image_rows, ignore_index=True)    #리스트에 있는 여러 df 세로로 합치기/ 인덱스는 0부터 다시 정렬  
            merged_df.to_csv(output_path, sep='\t', index=False)    #tsv파일로 저장 
            print(f"Saved: {output_path}")
'''
import os
import pandas as pd
from glob import glob

# TSV 파일들이 있는 상위 디렉토리
tsv_dir = '/nas/research/03-Neural_decoding/3-bids/derivatives/b4_roi_zscore'

# 모든 *_task-image_events.tsv 파일 찾기
tsv_files = glob(os.path.join(tsv_dir, 'sub-*', 'func', '*_task-image_events.tsv'))

for tsv_path in tsv_files:
    try:
        df = pd.read_csv(tsv_path, sep='\t')

        if 'train' in df.columns:
            df = df.drop(columns=['train'])
            df.to_csv(tsv_path, sep='\t', index=False)
            print(f"Removed 'train' column: {tsv_path}")
        else:
            print(f"No 'train' column to remove in: {tsv_path}")

    except Exception as e:
        print(f"Error processing {tsv_path}: {e}")
