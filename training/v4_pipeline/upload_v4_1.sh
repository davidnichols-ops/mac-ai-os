#!/bin/bash
# Upload v4.1 merged model + adapters to HuggingFace
set -e
if [ -z "$HF_TOKEN" ]; then
  echo "HF_TOKEN must be set in the environment" >&2
  exit 1
fi

MERGED_DIR="/dev/shm/merged_model"
SFT_DIR="/root/training/models/sft_adapter"
DPO_DIR="/root/training/models/dpo_adapter"

echo "=== Uploading v4.1 to HuggingFace ==="

python3 -c "
from huggingface_hub import HfApi
api = HfApi(token='$HF_TOKEN')

# Upload merged model (overwrite v4 repo)
print('Uploading merged model to davidnichols-ops/adaptive-operator-v4...')
api.upload_folder(
    folder_path='$MERGED_DIR',
    repo_id='davidnichols-ops/adaptive-operator-v4',
    repo_type='model',
)
print('Merged model uploaded!')

# Upload SFT adapter
print('Uploading SFT adapter...')
api.upload_folder(
    folder_path='$SFT_DIR',
    repo_id='davidnichols-ops/adaptive-operator-v4',
    repo_type='model',
    path_in_repo='sft_adapter',
)
print('SFT adapter uploaded!')

# Upload DPO adapter
print('Uploading DPO adapter...')
api.upload_folder(
    folder_path='$DPO_DIR',
    repo_id='davidnichols-ops/adaptive-operator-v4',
    repo_type='model',
    path_in_repo='dpo_adapter',
)
print('DPO adapter uploaded!')
print('All uploads complete!')
"
