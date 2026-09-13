#!/usr/bin/env bash
set -euo pipefail
ENV_NAME="${EXPEL_CONDA_ENV:-expel-qwen}"
if ! conda env list | awk '{print $1}' | grep -Fxq "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" python=3.9.17
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
python -m pip install --upgrade 'pip<25'
python -m pip install -r requirements.txt
python -m pip install \
  'numpy==1.26.4' \
  'pandas==1.5.3' \
  'huggingface_hub==0.25.2' \
  'sentence_transformers==2.2.2' \
  'faiss-cpu==1.7.4'
python -m pip install 'alfworld[full]'
python -m pip check
python scripts/write_environment_audit.py --output artifacts/expel_qwen3_4b/environment_audit.txt
