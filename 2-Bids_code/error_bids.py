import os

def delete_sbref_files(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if '_sbref' in filename:
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"삭제됨: {file_path}")
                except Exception as e:
                    print(f"삭제 실패: {file_path} - {e}")


delete_sbref_files('/nas/research/03-Neural_decoding/3-bids/raw_data/sub-08')
