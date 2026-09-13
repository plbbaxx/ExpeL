#!/usr/bin/env bash
set -euo pipefail
echo "Qwen3-4B candidates:"
find /mnt/disk2/caoxue/models /mnt/disk1/caoxue/bench/models -maxdepth 3 -type f -name config.json -printf '%h\n' 2>/dev/null | grep -i 'qwen3.*4b' || true
echo "all-mpnet-base-v2 candidates:"
find /mnt/disk2/caoxue/models /mnt/disk1/caoxue/bench/models -maxdepth 3 -type f -name config.json -printf '%h\n' 2>/dev/null | grep -i 'all-mpnet-base-v2' || true
