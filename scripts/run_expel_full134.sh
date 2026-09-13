#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

source scripts/expel_qwen_runtime.sh
python scripts/prepare_alfworld_manifests.py

BASE="artifacts/expel_qwen3_4b/full134"
MANIFEST="data/alfworld/manifests/full134.json"
STAGE="${STAGE:-all}"
mkdir -p "$BASE"

run_gather() {
  export EXPEL_LLM_USAGE_LOG="$ROOT_DIR/$BASE/gather_llm_usage.jsonl"
  python train.py \
    benchmark=alfworld \
    agent=expel_qwen \
    benchmark.task_file="$MANIFEST" \
    log_dir="$BASE" \
    run_name=experience_gathering \
    testing=false \
    resume="${RESUME:-false}" \
    2>&1 | tee "$BASE/gather_console.log"
}

run_insights() {
  export EXPEL_LLM_USAGE_LOG="$ROOT_DIR/$BASE/insight_llm_usage.jsonl"
  python insight_extraction.py \
    benchmark=alfworld \
    agent=expel_qwen \
    benchmark.task_file="$MANIFEST" \
    log_dir="$BASE" \
    load_run_name=experience_gathering \
    run_name=insights_qwen3_4b \
    testing=false \
    resume="${INSIGHT_RESUME:-false}" \
    2>&1 | tee "$BASE/insight_console.log"
}

run_eval() {
  export EXPEL_LLM_USAGE_LOG="$ROOT_DIR/$BASE/eval_llm_usage.jsonl"
  python eval.py \
    benchmark=alfworld \
    agent=expel_qwen \
    benchmark.task_file="$MANIFEST" \
    benchmark.eval_configs.k_folds=2 \
    log_dir="$BASE" \
    load_run_name=extracted_insights/insights_qwen3_4b \
    run_name=eval_full_expel \
    no_rules=false \
    agent.fewshot_strategy=task_similarity \
    testing=false \
    resume="${EVAL_RESUME:-false}" \
    2>&1 | tee "$BASE/eval_full_expel_console.log"
}

case "$STAGE" in
  gather) run_gather ;;
  insights) run_insights ;;
  eval) run_eval ;;
  all)
    run_gather
    run_insights
    run_eval
    ;;
  *)
    echo "Unknown STAGE=$STAGE (expected: gather, insights, eval, all)" >&2
    exit 2
    ;;
esac
