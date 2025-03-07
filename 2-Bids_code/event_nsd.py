import os
import pandas as pd

def process_events_files(bids_dir, subjects):
    for subject in subjects:  # 특정 subject 디렉토리만 처리
        subject_dir = os.path.join(bids_dir, subject)
        if not os.path.exists(subject_dir):
            print(f"Directory not found: {subject_dir}")
            continue
        
        # Traverse the subject's directory tree
        for root, dirs, files in os.walk(subject_dir):
            for file in files:
                if file.endswith('_events.tsv'):
                    file_path = os.path.join(root, file)
                    print(f"Processing file: {file_path}")
                    
                    # Read the TSV file
                    df = pd.read_csv(file_path, sep='\t')

                    # 73k_id -> image로 이름 바꾸기기
                    if '73k_id' in df.columns:
                        df.rename(columns={'73k_id': 'image'}, inplace=True)
                        df['image'] = df['image'].apply(lambda x: f"coco2017_{x}" if pd.notnull(x) else x)
                    
                    # 컬럼 순서 변경 (지정한 컬럼을 앞으로, 나머지는 그대로 뒤에 유지)
                    columns_order = ['onset', 'duration', 'trial_number', 'image', 'response_time']
                    remaining_columns = [col for col in df.columns if col not in columns_order]
                    df = df[columns_order + remaining_columns]

                    # Save the modified file back
                    df.to_csv(file_path, sep='\t', index=False)
                    print(f"Updated file saved: {file_path}")

# Define the BIDS directory and subjects to process
bids_directory_path = "/nas-tmp/research/03-Neural_decoding/3-bids"
subjects_to_process = [f"sub-0{i}" for i in range(1, 9)]  # ['sub-01', 'sub-02', ..., 'sub-08']

process_events_files(bids_directory_path, subjects_to_process)