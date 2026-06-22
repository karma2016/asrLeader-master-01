#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BASE_IMAGE="${BASE_IMAGE:-meeting-asr:npu-new-oldbase}"
IMAGE_TAG="${IMAGE_TAG:-funasr-leader-asr:npu-prod-oldbase-20260610}"
NERDCTL_NAMESPACE="${NERDCTL_NAMESPACE:-k8s.io}"
BUILD_CONTAINER="${BUILD_CONTAINER:-asr-prod-oldbase-build}"
PUSH_IMAGE="${PUSH_IMAGE:-false}"

run_nerdctl() {
  nerdctl -n "${NERDCTL_NAMESPACE}" "$@"
}

cleanup() {
  run_nerdctl rm -f "${BUILD_CONTAINER}" >/dev/null 2>&1 || true
}

trap cleanup EXIT

cd "${ROOT}"

if ! command -v nerdctl >/dev/null 2>&1; then
  echo "nerdctl is required on the production Ascend node" >&2
  exit 1
fi

if ! run_nerdctl image inspect "${BASE_IMAGE}" >/dev/null 2>&1; then
  echo "missing base image in namespace ${NERDCTL_NAMESPACE}: ${BASE_IMAGE}" >&2
  echo "Run this on the worker that already has the production image." >&2
  exit 1
fi

if ! grep -q "torch_npu ready: devices=" "${ROOT}/docker/entrypoint.npu.sh"; then
  echo "entrypoint is missing the verified torch_npu preflight" >&2
  exit 1
fi

if ! grep -q "/usr/local/Ascend/nnal/atb/8.0.0/atb/cxx_abi_1/lib" "${ROOT}/docker/entrypoint.npu.sh"; then
  echo "entrypoint is missing the verified production LD_LIBRARY_PATH" >&2
  exit 1
fi

QWEN_DIR="${ROOT}/models/Qwen2.5-1.5B-Instruct"
QWEN_WEIGHTS="${QWEN_DIR}/model.safetensors"
if [ ! -f "${QWEN_WEIGHTS}" ]; then
  echo "missing bundled Qwen weights: ${QWEN_WEIGHTS}" >&2
  exit 1
fi

qwen_size="$(stat -c '%s' "${QWEN_WEIGHTS}")"
if [ "${qwen_size}" -lt 1000000000 ]; then
  echo "Qwen model.safetensors is too small: ${qwen_size} bytes" >&2
  exit 1
fi

echo "Checking production base package versions"
run_nerdctl run --rm \
  --net none \
  --entrypoint python \
  "${BASE_IMAGE}" \
  -c 'import importlib.metadata as m, sys
expected = {
    "funasr": "1.3.1",
    "modelscope": "1.36.0",
    "torch": "2.1.0",
    "torch-npu": "2.1.0.post10",
    "fastapi": "0.136.0",
    "pydantic": "2.9.2",
    "transformers": "4.44.0",
}
assert sys.version_info[:2] == (3, 11), sys.version
for name, wanted in expected.items():
    actual = m.version(name)
    print(f"{name}={actual}")
    assert actual == wanted, f"{name}: expected {wanted}, got {actual}"'

cleanup

echo "Base image : ${BASE_IMAGE}"
echo "Target tag : ${IMAGE_TAG}"
echo "Namespace  : ${NERDCTL_NAMESPACE}"

run_nerdctl run -d \
  --net none \
  --name "${BUILD_CONTAINER}" \
  --entrypoint /bin/sh \
  "${BASE_IMAGE}" \
  -c "sleep 86400" >/dev/null

run_nerdctl exec "${BUILD_CONTAINER}" mkdir -p /app/app /app/models /data
run_nerdctl cp "${ROOT}/app/." "${BUILD_CONTAINER}:/app/app"
run_nerdctl cp "${ROOT}/docker/entrypoint.npu.sh" "${BUILD_CONTAINER}:/usr/local/bin/asr-npu-entrypoint.sh"
run_nerdctl exec "${BUILD_CONTAINER}" chmod 755 /usr/local/bin/asr-npu-entrypoint.sh

echo "Copying bundled Qwen model (${qwen_size} bytes)"
run_nerdctl cp "${QWEN_DIR}" "${BUILD_CONTAINER}:/app/models/Qwen2.5-1.5B-Instruct"

run_nerdctl exec "${BUILD_CONTAINER}" python -m compileall -q /app/app

run_nerdctl commit \
  --change 'ENTRYPOINT ["/usr/local/bin/asr-npu-entrypoint.sh"]' \
  --change 'CMD ["python","/app/app/server.py"]' \
  "${BUILD_CONTAINER}" \
  "${IMAGE_TAG}"

echo
echo "Built image:"
run_nerdctl image inspect "${IMAGE_TAG}" | grep -A 3 -E '"Entrypoint"|"Cmd"' || true

if [ "${PUSH_IMAGE}" = "true" ]; then
  echo
  echo "Pushing image: ${IMAGE_TAG}"
  run_nerdctl push "${IMAGE_TAG}"
fi

echo
echo "Build complete. Follow npu-production-update-runbook.md on master."
echo "Route traffic to dream-acr-new before replacing the NPU deployment."
echo "Do not use a rolling update when only one NPU is available."
