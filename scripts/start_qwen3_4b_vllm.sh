#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_PATH="${QWEN_MODEL_PATH:-/mnt/disk2/caoxue/models/Qwen3-4B-Instruct-2507}"
SERVED_NAME="${QWEN_SERVED_NAME:-Qwen3-4B-Instruct-2507}"
GPU_ID="${GPU_ID:-0}"
PORT="${VLLM_PORT:-18000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-16384}"
SESSION="${VLLM_TMUX_SESSION:-expel-qwen3-4b-18000}"
LOG_DIR="$ROOT_DIR/artifacts/expel_qwen3_4b/model_server"
mkdir -p "$LOG_DIR"
test -f "$MODEL_PATH/config.json" || { echo "Missing local Qwen checkpoint: $MODEL_PATH"; exit 1; }
if curl --fail --silent "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1; then
  echo "Port $PORT already has a healthy OpenAI-compatible server; no new tmux session created."
  exit 0
fi
tmux has-session -t "$SESSION" 2>/dev/null && { echo "tmux session already exists: $SESSION"; exit 1; }
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT_DIR' && CUDA_VISIBLE_DEVICES='$GPU_ID' vllm serve '$MODEL_PATH' --served-model-name '$SERVED_NAME' --host 127.0.0.1 --port '$PORT' --dtype bfloat16 --tensor-parallel-size 1 --gpu-memory-utilization 0.75 --max-model-len '$MAX_MODEL_LEN' --generation-config vllm 2>&1 | tee '$LOG_DIR/vllm-$PORT.log'"
echo "Started $SESSION. Health: curl http://127.0.0.1:$PORT/v1/models"
echo "Logs: tmux capture-pane -pt $SESSION -S -100"
