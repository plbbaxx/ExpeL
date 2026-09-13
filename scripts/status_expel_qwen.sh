#!/usr/bin/env bash
set -euo pipefail
BASE="${1:-artifacts/expel_qwen3_4b/pilot30}"
echo "Processes:"
ps -ef | grep -E '[p]ython (train|eval|insight_extraction)\.py' || true
echo "Gathered task checkpoints:"
if [[ -f "$BASE/alfworld/expel/experience_gathering.txt" ]]; then
  grep -c '^TASK [0-9]' "$BASE/alfworld/expel/experience_gathering.txt" || true
else
  echo 0
fi
echo "Evaluation files:"
find "$BASE/alfworld/expel/eval" -maxdepth 1 -type f 2>/dev/null | sort || true
echo "Recent console output:"
tail -n 30 "$BASE/gather_console.log" 2>/dev/null || true
