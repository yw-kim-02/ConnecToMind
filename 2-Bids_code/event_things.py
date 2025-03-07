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
                    df = pd.read_csv(file_path, sep="\t")

                    # Rename file_path to image and modify values
                    if 'file_path' in df.columns:
                        df.rename(columns={'file_path': 'image'}, inplace=True)
                        df['image'] = df['image'].apply(lambda x: f"things_{os.path.basename(x)}" if pd.notnull(x) else x) # os.path.basename(x)는 경로에서 파일명만 추출하는 코드

                    # Keep only specified columns in correct order
                    columns_order = ['onset', 'duration', 'image']
                    df = df[[col for col in columns_order if col in df.columns]]

                    # Save the modified file back
                    df.to_csv(file_path, sep="\t", index=False)
                    print(f"Updated file saved: {file_path}")

# Define the BIDS directory and subjects to process
bids_directory_path = "/nas-tmp/research/03-Neural_decoding/3-bids"
subjects_to_process = [f"sub-0{i}" if i < 10 else f"sub-{i}" for i in range(18, 21)]

process_events_files(bids_directory_path, subjects_to_process)
