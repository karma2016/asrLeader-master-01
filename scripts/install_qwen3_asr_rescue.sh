#!/usr/bin/env bash
set -euo pipefail

runtime_dir="${ASR_QWEN_RESCUE_RUNTIME_PATH:-/data/qwen_asr_runtime}"
model_dir="${ASR_QWEN_RESCUE_MODEL:-/data/qwen3-asr-0.6b}"
model_id="${ASR_QWEN_RESCUE_MODEL_ALIAS:-Qwen/Qwen3-ASR-0.6B}"

python -m pip config set global.index-url "${PIP_INDEX_URL:-https://mirrors.aliyun.com/pypi/simple/}"
python -m pip config set global.trusted-host "${PIP_TRUSTED_HOST:-mirrors.aliyun.com}"

mkdir -p "${runtime_dir}"
python -m pip install --no-deps qwen-asr==0.0.6 --target "${runtime_dir}"
python -m pip install \
  transformers==4.57.6 \
  qwen-omni-utils==0.0.9 \
  nagisa==0.2.11 \
  soynlp==0.0.493 \
  "sox>=1.5.0" \
  "huggingface-hub>=0.34.0,<1.0" \
  --target "${runtime_dir}"

if [ ! -f "${model_dir}/model.safetensors" ]; then
  mkdir -p "${model_dir}"
  modelscope download --model "${model_id}" --local_dir "${model_dir}"
fi

PYTHONPATH="${runtime_dir}" python - <<'PY'
from qwen_asr import Qwen3ASRModel
print("Qwen3-ASR runtime import OK:", Qwen3ASRModel)
PY
