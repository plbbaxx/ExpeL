#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
source scripts/expel_qwen_runtime.sh
python scripts/prepare_alfworld_manifests.py
OUT="artifacts/expel_qwen3_4b/smoke/react10"; mkdir -p "$OUT"
export EXPEL_LLM_USAGE_LOG="$ROOT_DIR/$OUT/llm_usage.jsonl"
python train.py benchmark=alfworld agent=react_qwen benchmark.task_file=data/alfworld/manifests/smoke10.json \
  log_dir="$OUT" run_name=react_smoke10 testing=false resume="${RESUME:-false}" 2>&1 | tee "$OUT/console.log"
