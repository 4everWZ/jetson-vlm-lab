#!/usr/bin/env bash

llama_cpp_runtime_gate_safe_name() {
  printf '%s' "${1:?image required}" | tr '/:@' '___' | tr -cd 'A-Za-z0-9_.-'
}

require_llama_cpp_multimodal_runtime() {
  local image="${1:?image required}"
  local docker_gpu_args="${2-}"
  local llama_server_cmd="${3-}"

  if [[ "${JETSON_DRY_RUN:-0}" == "1" ]]; then
    return 0
  fi

  local gate_dir repo_root python_bin probe_output safe_image
  gate_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  repo_root="$(cd "${gate_dir}/../.." && pwd)"
  python_bin="${PYTHON_BIN:-python3}"
  if [[ -n "${LLAMA_CPP_RUNTIME_PROBE_OUTPUT:-}" ]]; then
    probe_output="${LLAMA_CPP_RUNTIME_PROBE_OUTPUT}"
  else
    safe_image="$(llama_cpp_runtime_gate_safe_name "${image}")"
    probe_output="${repo_root}/outputs/jetson_inspect/llama_cpp_runtime_probe-${safe_image}.json"
  fi

  local -a probe_cmd=(
    "${python_bin}"
    -m edge_vlm.llama_cpp_runtime
    probe-image
    --image "${image}"
    --output "${probe_output}"
    --docker-gpu-args "${docker_gpu_args}"
  )
  if [[ -n "${llama_server_cmd}" ]]; then
    probe_cmd+=(--llama-server-cmd "${llama_server_cmd}")
  fi

  if ! PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${probe_cmd[@]}" >/dev/null; then
    echo "runtime_probe_failed: unable to probe llama.cpp image ${image}; see ${probe_output}" >&2
    return 2
  fi

  local multimodal_ready
  if ! multimodal_ready="$("${python_bin}" - "${probe_output}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as handle:
    artifact = json.load(handle)

ready = artifact.get("multimodal_ready") is True
if not ready:
    markers = ",".join(artifact.get("llama_server_multimodal_markers") or [])
    print(
        "runtime_probe_detail: "
        f"llama_server_path={artifact.get('llama_server_path')}; "
        f"llama_server_help_ok={artifact.get('llama_server_help_ok')}; "
        f"llama_server_supports_mmproj={artifact.get('llama_server_supports_mmproj')}; "
        f"llama_server_multimodal_markers={markers}",
        file=sys.stderr,
    )
print("1" if ready else "0")
PY
  )"; then
    echo "runtime_probe_failed: invalid llama.cpp runtime probe artifact ${probe_output}" >&2
    return 2
  fi

  if [[ "${multimodal_ready}" != "1" ]]; then
    echo "runtime_missing_mmproj_support: image ${image} did not expose exact --mmproj in llama-server --help; see ${probe_output}" >&2
    return 2
  fi
}
