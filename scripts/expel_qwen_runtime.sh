#!/usr/bin/env bash
export OPENAI_API_KEY="${OPENAI_API_KEY:-local}"
export EXPEL_OPENAI_API_BASE="${EXPEL_OPENAI_API_BASE:-http://127.0.0.1:18000/v1}"
export EXPEL_TEMPERATURE="${EXPEL_TEMPERATURE:-0}"
export EXPEL_TOP_P="${EXPEL_TOP_P:-1}"
export EXPEL_MAX_TOKENS="${EXPEL_MAX_TOKENS:-512}"
export EXPEL_SEED="${EXPEL_SEED:-42}"
export EXPEL_EMBEDDER_PATH="${EXPEL_EMBEDDER_PATH:-/mnt/disk2/caoxue/models/all-mpnet-base-v2}"
export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
export no_proxy="127.0.0.1,localhost${no_proxy:+,$no_proxy}"
test -d "$EXPEL_EMBEDDER_PATH" || { echo "Missing local official embedder: $EXPEL_EMBEDDER_PATH"; return 1 2>/dev/null || exit 1; }
curl --fail --silent --show-error "$EXPEL_OPENAI_API_BASE/models" >/dev/null || { echo "Qwen server is not healthy: $EXPEL_OPENAI_API_BASE"; return 1 2>/dev/null || exit 1; }
