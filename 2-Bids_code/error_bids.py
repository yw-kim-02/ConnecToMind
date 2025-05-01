import os

targets = ['_recording-cardiac_physio.tsv', '_recording-respiratory_physio.tsv']

root_dir = '/nas/research/03-Neural_decoding/3-bids/raw_data/sub-08'

deleted_files = []
for dirpath, dirnames, filenames in os.walk(root_dir):
    for filename in filenames:
        if any(target in filename for target in targets):
            full_path = os.path.join(dirpath, filename)
            try:
                os.remove(full_path)
                deleted_files.append(full_path)
            except Exception as e:
                print(f"failed to delete: {full_path} ({e})")

print(f"\nnumber of deleted files: {len(deleted_files)}:")
for f in deleted_files:
    print(f)
