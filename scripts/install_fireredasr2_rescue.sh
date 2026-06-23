#!/usr/bin/env bash
set -euo pipefail

runtime_dir="${ASR_FIRERED_RESCUE_RUNTIME_PATH:-/data/fireredasr2s_runtime}"
model_dir="${ASR_FIRERED_RESCUE_MODEL:-/data/fireredasr2-aed}"
repo_url="${ASR_FIRERED_RESCUE_REPO_URL:-https://github.com/FireRedTeam/FireRedASR2S.git}"
source_zip_url="${ASR_FIRERED_RESCUE_SOURCE_ZIP:-https://github.com/FireRedTeam/FireRedASR2S/archive/refs/heads/main.zip}"
model_id="${ASR_FIRERED_RESCUE_MODEL_ID:-xukaituo/FireRedASR2-AED}"

mkdir -p "$(dirname "${runtime_dir}")" "$(dirname "${model_dir}")"

if command -v git >/dev/null 2>&1; then
  if [ ! -d "${runtime_dir}/.git" ]; then
    rm -rf "${runtime_dir}"
    git clone --depth 1 "${repo_url}" "${runtime_dir}"
  else
    git -C "${runtime_dir}" pull --ff-only
  fi
else
  python - <<PY
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile
import io
import shutil

runtime = Path("${runtime_dir}")
if not (runtime / "fireredasr2s").is_dir():
    print("git not found; downloading FireRedASR2S source zip")
    data = urlopen("${source_zip_url}", timeout=120).read()
    tmp = runtime.with_suffix(".zip_extract")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True, exist_ok=True)
    with ZipFile(io.BytesIO(data)) as archive:
        archive.extractall(tmp)
    roots = [item for item in tmp.iterdir() if item.is_dir()]
    if not roots:
        raise RuntimeError("FireRedASR2S source zip had no root directory")
    shutil.rmtree(runtime, ignore_errors=True)
    shutil.move(str(roots[0]), str(runtime))
    shutil.rmtree(tmp, ignore_errors=True)
else:
    print(f"FireRedASR2S runtime already exists: {runtime}")
PY
fi

python -m pip install --no-cache-dir \
  cn2an==0.5.23 \
  kaldi_native_fbank==1.15 \
  textgrid==1.6.1 \
  "peft>=0.13.2"

python - <<PY
from pathlib import Path
from modelscope import snapshot_download

model_dir = Path("${model_dir}")
weight_patterns = ("*.pt", "*.pth", "*.pth.tar", "*.bin", "*.safetensors")
if not any(model_dir.glob(pattern) for pattern in weight_patterns):
    snapshot_download("${model_id}", local_dir=str(model_dir))
else:
    print(f"FireRedASR2 model already exists: {model_dir}")
PY

python - <<PY
from pathlib import Path
runtime = Path("${runtime_dir}")
model = Path("${model_dir}")
assert (runtime / "fireredasr2s").is_dir(), f"missing FireRedASR2S package: {runtime}"
assert model.is_dir(), f"missing FireRedASR2 model dir: {model}"
print(f"FireRedASR2 rescue runtime ready: {runtime}")
print(f"FireRedASR2 rescue model ready: {model}")
PY
