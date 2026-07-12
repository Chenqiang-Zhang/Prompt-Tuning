#!/usr/bin/env bash
# End-to-end reproduction driver: train a soft prompt per persona, generate on
# both the persona-eval and general-eval sets, then report distinct-1/2.
#
# Usage: scripts/run_experiment.sh <MODEL> <EPOCHS> <TAG> [DTYPE]
#   e.g. scripts/run_experiment.sh Qwen/Qwen2.5-0.5B 15 qwen05b
set -euo pipefail
cd "$(dirname "$0")/.."

MODEL="${1:-Qwen/Qwen2.5-0.5B}"
EPOCHS="${2:-15}"
TAG="${3:-run}"
DTYPE="${4:-float32}"
PERSONAS=(CP AN CT)
OUT="outputs/$TAG"
mkdir -p "$OUT"
export TOKENIZERS_PARALLELISM=false HF_HUB_DISABLE_PROGRESS_BARS=1

echo "=== MODEL=$MODEL EPOCHS=$EPOCHS TAG=$TAG DTYPE=$DTYPE ==="
for P in "${PERSONAS[@]}"; do
  echo ">>> train persona $P"
  python -u src/train.py --persona_dir "data/processed/persona_$P" \
    --model "$MODEL" --out "$OUT/persona_$P.pt" \
    --epochs "$EPOCHS" --dtype "$DTYPE"

  echo ">>> generate persona-eval $P"
  python -u src/generate.py --model "$MODEL" --prompt "$OUT/persona_$P.pt" \
    --dtype "$DTYPE" --eval_file "data/processed/persona_$P/eval_persona.jsonl" \
    --out "$OUT/gen_persona_$P.jsonl"

  echo ">>> generate general-eval $P"
  python -u src/generate.py --model "$MODEL" --prompt "$OUT/persona_$P.pt" \
    --dtype "$DTYPE" --eval_file "data/processed/eval_general.jsonl" \
    --out "$OUT/gen_general_$P.jsonl"
done

echo "=== DISTINCT (persona eval) ==="
python src/eval_distinct.py "$OUT"/gen_persona_*.jsonl
echo "=== DISTINCT (general eval) ==="
python src/eval_distinct.py "$OUT"/gen_general_*.jsonl
echo "DONE $TAG"
