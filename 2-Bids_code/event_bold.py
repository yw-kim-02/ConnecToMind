import os
import pandas as pd

def modify_imgname(row):
    img_name = row['ImgName'].lower()

    # "COCO_train2014_"를 "coco_train2014_"로 변경 (대소문자 변환)
    img_name = img_name.replace("coco_train2014_", "coco2014_")

    # rep_가 있는 경우 제거 (rep_coco 포함)
    img_name = img_name.replace("rep_", "")

    if img_name.endswith(".jpg"):
        img_name = img_name[:-4] 
    elif img_name.endswith(".jpeg"):
        img_name = img_name[:-5]  

    if row['ImgType'] == 'coco' or row['ImgType'] == 'rep_coco':
        return img_name
    elif row['ImgType'] == 'imagenet':
        return f"imagenet_{img_name}"
    elif row['ImgType'] == 'scenes':
        return f"scenes_{img_name}"
    elif row['ImgType'] == 'rep_imagenet':  # rep_imagenet일 경우
        return f"imagenet_{img_name}"
    elif row['ImgType'] == 'rep_scenes':  # rep_scenes일 경우
        return f"scenes_{img_name}"
    else:
        return img_name

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

                    # coco_, imagenet_, scene_ 으로 변경
                    if 'ImgName' in df.columns and 'ImgType' in df.columns:
                        df['ImgName'] = df.apply(modify_imgname, axis=1)
                    
                    # ImgName->image, RT->response_time
                    df.rename(columns={'ImgName': 'image', 'RT': 'response_time'}, inplace=True)

                    # 컬럼 순서 변경 (지정한 컬럼을 앞으로, 나머지는 기존 순서 유지)
                    columns_order = ['onset', 'duration', 'image', 'response_time']
                    remaining_columns = [col for col in df.columns if col not in columns_order]
                    df = df[columns_order + remaining_columns]

                    # Save the modified file back
                    df.to_csv(file_path, sep='\t', index=False)
                    print(f"Updated file saved: {file_path}")

# Define the BIDS directory and subjects to process
bids_directory_path = "/nas-tmp/research/03-Neural_decoding/3-bids"
subjects_to_process = [f"sub-0{i}" if i < 10 else f"sub-{i}" for i in range(9, 13)]

process_events_files(bids_directory_path, subjects_to_process)
