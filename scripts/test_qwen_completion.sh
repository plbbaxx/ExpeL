#!/usr/bin/env bash
set -euo pipefail
BASE_URL="${EXPEL_OPENAI_API_BASE:-http://127.0.0.1:18000/v1}"
curl --fail --silent --show-error "$BASE_URL/chat/completions" \
  -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer local' \
  -d '{"model":"Qwen3-4B-Instruct-2507","messages":[{"role":"user","content":"Output only: look"}],"temperature":0,"top_p":1,"max_tokens":8,"seed":42}'
