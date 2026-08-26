#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
set -a
source .env
set +a
exec uv run app slack start
