import os
import pandas as pd

def format_imagenet_name(value):
    """
    기존 imagenet 숫자 값을 "imagenet_n0xxxxx_xxxxx" 형식으로 변환
    """
    if pd.isnull(value) or value == "n/a":
        return value  # 값이 없거나 'n/a'이면 그대로 반환

    try:
        value = str(value)
        parts = value.split(".")  # 소수점 기준으로 분리
        if len(parts) == 2:
            class_id = parts[0]  # 앞부분 (클래스 ID)
            img_id = str(int(parts[1]))  # 뒤부분 (정수 변환하여 앞의 0 제거)
            return f"imagenet_n0{class_id}_{img_id}"

    except Exception as e:
        print(f"Error formatting imagenet name: {value} - {e}")
        return value  # 변환 실패 시 원래 값 유지

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

                    # Rename columns
                    df.rename(columns={'stim_id': 'image', 'trial_no': 'trial_number'}, inplace=True)

                    # Modify image column values
                    if 'image' in df.columns:
                        df['image'] = df['image'].apply(format_imagenet_name)

                    # 컬럼 순서 변경 (지정한 컬럼을 앞으로, 나머지는 기존 순서 유지)
                    columns_order = ['onset', 'duration', 'trial_number', 'image', 'response_time']
                    remaining_columns = [col for col in df.columns if col not in columns_order]
                    df = df[columns_order + remaining_columns]

                    # Save the modified file back
                    df.to_csv(file_path, sep="\t", index=False)
                    print(f"Updated file saved: {file_path}")

# Define the BIDS directory and subjects to process
bids_directory_path = "/nas-tmp/research/03-Neural_decoding/3-bids"
subjects_to_process = [f"sub-0{i}" if i < 10 else f"sub-{i}" for i in range(13, 18)]

process_events_files(bids_directory_path, subjects_to_process)
