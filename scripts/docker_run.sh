#!/usr/bin/env bash
# Run a command inside the prompt-tuning container with one GPU pinned.
# Builds the image on first use. Bind-mounts the repo, runs as the host user
# (so no root-owned files leak onto the shared host), and keeps the HF model
# cache inside the repo (./.hfcache) so downloads persist across runs.
#
# Usage:
#   scripts/docker_run.sh                              # interactive shell
#   scripts/docker_run.sh bash scripts/run_experiment.sh Qwen/Qwen2.5-3B 15 qwen3b bfloat16
#   GPU=4 scripts/docker_run.sh python -c "import torch; print(torch.cuda.is_available())"
#
# Env vars: GPU (default 3), IMAGE (default ptune:latest)
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-ptune:latest}"
GPU="${GPU:-3}"

if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo ">>> building $IMAGE (first run only) ..."
  docker build -t "$IMAGE" -f docker/Dockerfile .
fi

mkdir -p .hfcache outputs

TTY=""
[ -t 0 ] && TTY="-it"

exec docker run --rm $TTY \
  --gpus "device=${GPU}" \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace \
  -e HF_HOME=/workspace/.hfcache \
  -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
  -e TOKENIZERS_PARALLELISM=false \
  -v "$PWD":/workspace -w /workspace \
  "$IMAGE" "${@:-bash}"
