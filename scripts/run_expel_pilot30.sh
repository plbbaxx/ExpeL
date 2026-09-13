#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
source scripts/expel_qwen_runtime.sh
python scripts/prepare_alfworld_manifests.py
BASE="artifacts/expel_qwen3_4b/pilot30"; mkdir -p "$BASE"
export EXPEL_LLM_USAGE_LOG="$ROOT_DIR/$BASE/llm_usage.jsonl"
python train.py benchmark=alfworld agent=expel_qwen benchmark.task_file=data/alfworld/manifests/pilot30.json \
  log_dir="$BASE" run_name=experience_gathering testing=false resume="${RESUME:-false}" 2>&1 | tee "$BASE/gather_console.log"
python insight_extraction.py benchmark=alfworld agent=expel_qwen benchmark.task_file=data/alfworld/manifests/pilot30.json \
  log_dir="$BASE" load_run_name=experience_gathering run_name=insights_qwen3_4b agent.max_num_rules=10 \
  agent.success_critique_num=8 testing=false resume=false 2>&1 | tee "$BASE/insight_console.log"
