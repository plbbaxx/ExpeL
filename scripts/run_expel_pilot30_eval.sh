#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"; cd "$ROOT_DIR"
source scripts/expel_qwen_runtime.sh
BASE="artifacts/expel_qwen3_4b/pilot30"; mkdir -p "$BASE"
export EXPEL_LLM_USAGE_LOG="$ROOT_DIR/$BASE/eval_llm_usage.jsonl"
COMMON=(benchmark=alfworld agent=expel_qwen benchmark.task_file=data/alfworld/manifests/pilot30.json log_dir="$BASE" load_run_name=extracted_insights/insights_qwen3_4b testing=false resume="${RESUME:-false}" benchmark.eval_configs.k_folds=2)
python eval.py "${COMMON[@]}" run_name=eval_keep_only no_rules=true agent.fewshot_strategy=none 2>&1 | tee "$BASE/eval_keep_only_console.log"
python eval.py "${COMMON[@]}" run_name=eval_experience_only no_rules=true agent.fewshot_strategy=task_similarity 2>&1 | tee "$BASE/eval_experience_only_console.log"
python eval.py "${COMMON[@]}" run_name=eval_insights_only no_rules=false agent.fewshot_strategy=none 2>&1 | tee "$BASE/eval_insights_only_console.log"
python eval.py "${COMMON[@]}" run_name=eval_full_expel no_rules=false agent.fewshot_strategy=task_similarity 2>&1 | tee "$BASE/eval_full_expel_console.log"
python scripts/analyze_expel_transfer.py --base "$BASE"
