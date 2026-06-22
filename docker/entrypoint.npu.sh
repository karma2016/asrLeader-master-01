#!/usr/bin/env bash
set -e

prepend_env_path() {
  var_name="$1"
  new_path="$2"
  current_value="${!var_name:-}"

  if [ -z "$new_path" ] || [ ! -e "$new_path" ]; then
    return
  fi

  case ":${current_value}:" in
    *":${new_path}:"*) ;;
    *)
      if [ -n "$current_value" ]; then
        export "${var_name}=${new_path}:${current_value}"
      else
        export "${var_name}=${new_path}"
      fi
      ;;
  esac
}

export ASCEND_HOME_PATH="${ASCEND_HOME_PATH:-/usr/local/Ascend/ascend-toolkit/latest}"
export ASCEND_OPP_PATH="${ASCEND_OPP_PATH:-${ASCEND_HOME_PATH}/opp}"
export ASCEND_AICPU_PATH="${ASCEND_AICPU_PATH:-${ASCEND_HOME_PATH}}"
export TOOLCHAIN_HOME="${TOOLCHAIN_HOME:-${ASCEND_HOME_PATH}/toolkit}"

# Match the old working image behavior: source CANN before Python starts.
for env_file in \
  /usr/local/Ascend/ascend-toolkit/set_env.sh \
  "${ASCEND_HOME_PATH}/set_env.sh"; do
  if [ -f "$env_file" ]; then
    # shellcheck disable=SC1090
    . "$env_file"
    break
  fi
done

# Keep the exact library order from the working production old-base pod.
# Reconstructing this path from generic CANN directories can load an
# incompatible libacl_tdt_channel.so and fail with an undefined symbol.
export LD_LIBRARY_PATH="/usr/local/Ascend/atb/latest/atb/cxx_abi_1/lib:/usr/local/Ascend/atb/latest/atb/cxx_abi_1/examples:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64:/usr/local/Ascend/ascend-toolkit/latest/tools/aml/lib64/plugin:/usr/local/Ascend/ascend-toolkit/latest/lib64:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/opskernel:/usr/local/Ascend/ascend-toolkit/latest/lib64/plugin/nnengine:/usr/local/Ascend/ascend-toolkit/latest/opp/built-in/op_impl/ai_core/tbe/op_tiling/lib/linux/aarch64:/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:/usr/local/Ascend/ascend-toolkit/8.0.RC2/aarch64-linux/devlib/linux/aarch64:/usr/local/Ascend/nnal/atb/8.0.0/atb/cxx_abi_1/lib"
prepend_env_path PYTHONPATH "${ASCEND_HOME_PATH}/python/site-packages"
prepend_env_path PYTHONPATH "${ASCEND_OPP_PATH}/built-in/op_impl/ai_core/tbe"

# FunASR 1.3.1 in the working production image loads ASR models through the
# ModelScope registry/cache aliases. Resolving aliases to /app/models paths
# makes it fail with "is not registered".
export ASR_RESOLVE_LOCAL_MODELS="${ASR_RESOLVE_LOCAL_MODELS:-false}"

if [ "${POSTPROCESS_PRELOAD:-false}" = "true" ]; then
  postprocess_model_dir="${POSTPROCESS_MODEL_DIR:-}"
  if [ -n "$postprocess_model_dir" ] && [ ! -d "$postprocess_model_dir" ]; then
    export POSTPROCESS_PRELOAD=false
    echo "Disabled post-process preload: missing ${postprocess_model_dir}"
  fi
fi

# The old production pod worked with ASCEND_VISIBLE_DEVICES set to the physical
# NPU id from the device plugin. Keep that default and only normalize when
# explicitly requested.
normalize="${ASR_NORMALIZE_ASCEND_DEVICES:-false}"

visible_devices="${ASCEND_RT_VISIBLE_DEVICES:-${ASCEND_VISIBLE_DEVICES:-}}"

if [ "$normalize" != "false" ] && [ -n "$visible_devices" ]; then
  should_normalize=false
  if [ "$normalize" = "true" ]; then
    should_normalize=true
  fi

  if [ "$should_normalize" = "true" ] && [[ "$visible_devices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    original_devices="$visible_devices"
    IFS=',' read -r -a devices <<< "$original_devices"
    local_devices=""
    for index in "${!devices[@]}"; do
      if [ -n "$local_devices" ]; then
        local_devices="${local_devices},${index}"
      else
        local_devices="${index}"
      fi
    done

    export ASCEND_RT_VISIBLE_DEVICES="$local_devices"
    echo "Normalized Ascend runtime devices: ${original_devices} -> ${local_devices}"
  fi
fi

echo "ASR NPU runtime: ASR_DEVICE=${ASR_DEVICE:-} ASCEND_VISIBLE_DEVICES=${ASCEND_VISIBLE_DEVICES:-} ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES:-} ASR_RESOLVE_LOCAL_MODELS=${ASR_RESOLVE_LOCAL_MODELS:-} POSTPROCESS_PRELOAD=${POSTPROCESS_PRELOAD:-}"

wait_attempts="${ASR_NPU_READY_ATTEMPTS:-30}"
wait_seconds="${ASR_NPU_READY_INTERVAL_SECONDS:-2}"
echo "Waiting for torch_npu backend and Ascend device..."

for attempt in $(seq 1 "$wait_attempts"); do
  if python - <<'PY'
import torch
import torch_npu  # noqa: F401

if not hasattr(torch, "npu"):
    raise SystemExit("torch.npu backend is not registered")

count = torch.npu.device_count()
if count < 1:
    raise SystemExit("no Ascend NPU is visible")

print(f"torch_npu ready: devices={count}")
PY
  then
    exec "$@"
  fi

  if [ "$attempt" -lt "$wait_attempts" ]; then
    echo "NPU is not ready yet (${attempt}/${wait_attempts}); retrying in ${wait_seconds}s..."
    sleep "$wait_seconds"
  fi
done

echo "NPU preflight failed after ${wait_attempts} attempts" >&2
exit 1
