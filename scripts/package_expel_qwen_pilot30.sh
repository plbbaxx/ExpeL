#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
BASE="artifacts/expel_qwen3_4b/pilot30"; ARCHIVE="${1:-expel_qwen3_4b_alfworld_pilot30_results.tar.gz}"
test -f "$BASE/summary.json" || { echo "Missing completed summary: $BASE/summary.json"; exit 1; }
tar -czf "$ARCHIVE" "$BASE" artifacts/expel_qwen3_4b/environment_audit.txt \
  data/alfworld/manifests configs/agent/expel_qwen.yaml configs/agent/react_qwen.yaml \
  models/llm.py agent/expel.py prompts/alfworld.py scripts
echo "$ARCHIVE"
