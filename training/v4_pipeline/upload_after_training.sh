#!/bin/bash
# Post-training upload script — runs after vast_train.py completes
# Uploads model artifacts to:
#   1. HuggingFace Hub (if HF_TOKEN set)
#   2. Local Mac via SCP (if SSH key available)

set -e

MODELS_DIR="/root/training/models"
MERGED_DIR="$MODELS_DIR/merged"
SFT_DIR="$MODELS_DIR/sft_adapter"
DPO_DIR="$MODELS_DIR/dpo_adapter"

echo "=== Post-Training Upload ==="
echo "Waiting for training to complete..."

# Wait for the merged model to appear (training completion signal)
while [ ! -d "$MERGED_DIR" ] || [ -z "$(ls -A $MERGED_DIR 2>/dev/null)" ]; do
    sleep 60
    echo "  Still training... (checking for $MERGED_DIR)"
done

echo "Training complete! Model found at $MERGED_DIR"
echo ""

# Show what we have
echo "=== Model artifacts ==="
ls -lhR $MODELS_DIR/
echo ""

# Check sizes
echo "=== Sizes ==="
du -sh $SFT_DIR $DPO_DIR $MERGED_DIR 2>/dev/null
echo ""

# Upload to HuggingFace if token is set
if [ -n "$HF_TOKEN" ]; then
    echo "=== Uploading to HuggingFace ==="
    python3 -c "
from huggingface_hub import HfApi, create_repo
import os

api = HfApi(token=os.environ['HF_TOKEN'])
repo_id = 'davidnichols-ops/adaptive-operator-v4'

try:
    create_repo(repo_id, repo_type='model', exist_ok=True, token=os.environ['HF_TOKEN'])
except Exception as e:
    print(f'Repo exists or error: {e}')

# Upload merged model
print('Uploading merged model...')
api.upload_folder(
    folder_path='$MERGED_DIR',
    repo_id=repo_id,
    repo_type='model',
)
print('Merged model uploaded!')

# Upload SFT adapter
print('Uploading SFT adapter...')
api.upload_folder(
    folder_path='$SFT_DIR',
    path_in_repo='sft_adapter',
    repo_id=repo_id,
    repo_type='model',
)
print('SFT adapter uploaded!')

# Upload DPO adapter
print('Uploading DPO adapter...')
api.upload_folder(
    folder_path='$DPO_DIR',
    path_in_repo='dpo_adapter',
    repo_id=repo_id,
    repo_type='model',
)
print('DPO adapter uploaded!')
print('All artifacts uploaded to HuggingFace!')
"
else
    echo "HF_TOKEN not set — skipping HuggingFace upload"
fi

echo ""
echo "=== Upload complete ==="
echo "Model artifacts are at: $MODELS_DIR"
echo "Remember to download them before destroying the instance!"
