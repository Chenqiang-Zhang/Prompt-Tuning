#!/usr/bin/env bash
# One-shot setup on a fresh CUDA server: install deps, fetch data, prepare splits.
# Usage: bash scripts/setup_cloud.sh
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== 1. Python deps ==="
pip install -U pip
# Install a CUDA torch wheel matched to the driver. cu121 wheels need driver
# >=525, so they run on common lab drivers (e.g. 535 / CUDA 12.2). Override the
# index with TORCH_INDEX_URL if your driver needs a different build.
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu121}"
if python -c "import torch" 2>/dev/null; then
  echo "torch already present, skipping torch install"
else
  pip install torch --index-url "$TORCH_INDEX_URL"
fi
pip install -r requirements.txt   # torch>=2.3 already satisfied by the line above

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
