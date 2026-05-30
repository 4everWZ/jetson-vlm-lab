#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"
remote_sync="${JETSON_REMOTE_SYNC:-1}"
remote_pythonpath="${JETSON_REMOTE_PYTHONPATH:-src}"
llama_cpp_image="${JETSON_REMOTE_LLAMA_CPP_IMAGE:-ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <edge_vlm.jetson_sweep args...>" >&2
  exit 2
fi

if [[ "${remote_sync}" == "1" ]]; then
  "${remote_exec}" git pull --ff-only
elif [[ "${remote_sync}" != "0" ]]; then
  echo "JETSON_REMOTE_SYNC must be 0 or 1." >&2
  exit 2
fi

exec "${remote_exec}" \
  env \
  "LLAMA_CPP_DOCKER_IMAGE=${llama_cpp_image}" \
  "PYTHONPATH=${remote_pythonpath}" \
  bash scripts/jetson/run_optimization_sweep.sh \
  "$@"
