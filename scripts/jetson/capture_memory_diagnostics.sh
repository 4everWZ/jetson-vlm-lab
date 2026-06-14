#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${JETSON_ENV_FILE:-${repo_root}/.env.jetson}"
python_bin="${PYTHON_BIN:-python3}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
output="${JETSON_MEMORY_DIAGNOSTICS_OUTPUT:-${repo_root}/outputs/jetson_inspect/memory-diagnostics-${timestamp}.json}"

. "${repo_root}/scripts/jetson/env_file.sh"
jetson_load_env_file "${env_file}"

cd "${repo_root}"
export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

exec "${python_bin}" -m edge_vlm.jetson_memory_diagnostics --output "${output}" "$@"
