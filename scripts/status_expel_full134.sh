#!/usr/bin/env bash
set -euo pipefail

BASE="${1:-artifacts/expel_qwen3_4b/full134}"

for log in gather_console.log insight_console.log eval_full_expel_console.log; do
  path="$BASE/$log"
  if [[ -f "$path" ]]; then
    echo "$path"
    tail -n 8 "$path"
  else
    echo "$path: not started"
  fi
done

find "$BASE" -type f \
  \( -name 'experience_gathering.pkl' -o -name 'insights_qwen3_4b.pkl' -o -name 'eval_full_expel.pkl' \) \
  -printf '%p %k KB\n' 2>/dev/null || true
