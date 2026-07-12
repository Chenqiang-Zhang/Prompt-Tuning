#!/usr/bin/env bash
# One-shot setup on a fresh CUDA server: install deps, fetch data, prepare splits.
# Usage: bash scripts/setup_cloud.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Python deps ==="
# On a CUDA host, the default PyPI torch wheel already bundles CUDA.
pip install -U pip
pip install -r requirements.txt

echo "=== 2. Sanity: GPU visible? ==="
python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no GPU")
PY

echo "=== 3. Download RealPersonaChat v1.0.0 (public, 24MB) ==="
mkdir -p data
if [ ! -d data/real-persona-chat-1.0.0 ]; then
  curl -sL -o data/rpc.zip \
    https://github.com/nu-dialogue/real-persona-chat/archive/refs/tags/v1.0.0.zip
  (cd data && unzip -q -o rpc.zip)
fi

echo "=== 4. Prepare splits (top-3 personas, 525 pairs each, 1:1 general) ==="
python src/prepare_data.py --top_personas 3 --general_ratio 1.0 --max_pairs_per_persona 525

echo
echo "Setup done. Run the experiment (bfloat16 recommended on CUDA):"
echo "  scripts/run_experiment.sh Qwen/Qwen2.5-0.5B 15 qwen05b bfloat16"
echo "  scripts/run_experiment.sh Qwen/Qwen2.5-3B   15 qwen3b  bfloat16"
