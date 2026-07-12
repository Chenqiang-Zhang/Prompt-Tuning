#!/usr/bin/env bash
# Download RealPersonaChat and build the paper's splits.
# Uses only the Python standard library (curl + prepare_data.py), so it runs on
# the bare host without any pip install / container.
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PYTHON:-python3}"

mkdir -p data
if [ ! -d data/real-persona-chat-1.0.0 ]; then
  echo ">>> download RealPersonaChat v1.0.0 (public, 24MB)"
  curl -sL -o data/rpc.zip \
    https://github.com/nu-dialogue/real-persona-chat/archive/refs/tags/v1.0.0.zip
  # unzip via python stdlib (no `unzip` binary needed on minimal hosts)
  "$PY" -c "import zipfile; zipfile.ZipFile('data/rpc.zip').extractall('data')"
fi

echo ">>> prepare splits (top-3 personas, 525 pairs each, 1:1 general)"
"$PY" src/prepare_data.py --top_personas 3 --general_ratio 1.0 --max_pairs_per_persona 525
echo ">>> done"
