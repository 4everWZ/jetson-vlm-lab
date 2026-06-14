#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${JETSON_ENV_FILE:-${repo_root}/.env.jetson}"
access_check="${JETSON_REMOTE_ACCESS_CHECK:-${repo_root}/scripts/jetson/check_remote_access.sh}"
remote_probe="${JETSON_REMOTE_PROBE:-${repo_root}/scripts/jetson/remote_probe.sh}"
remote_sweep="${JETSON_REMOTE_SWEEP:-${repo_root}/scripts/jetson/run_remote_optimization_sweep.sh}"
run_prefix="${JETSON_REMOTE_SMOKE_RUN_PREFIX:-remote-preflight-smoke-$(date -u +%Y%m%dT%H%M%SZ)}"
variant="${JETSON_REMOTE_SMOKE_VARIANT:-gemma-q4-baseline-gpu12-b512-u512-kvq8}"
skip_access_check="${JETSON_REMOTE_SMOKE_SKIP_ACCESS_CHECK:-0}"

if [[ "${skip_access_check}" != "0" && "${skip_access_check}" != "1" ]]; then
  echo "JETSON_REMOTE_SMOKE_SKIP_ACCESS_CHECK must be 0 or 1." >&2
  exit 2
fi

if [[ "${skip_access_check}" == "0" ]]; then
  JETSON_ENV_FILE="${env_file}" "${access_check}"
fi

JETSON_ENV_FILE="${env_file}" "${remote_probe}"

sweep_access_preflight="${JETSON_REMOTE_ACCESS_PREFLIGHT:-1}"
if [[ "${skip_access_check}" == "1" && -z "${JETSON_REMOTE_ACCESS_PREFLIGHT:-}" ]]; then
  sweep_access_preflight="0"
fi

JETSON_ENV_FILE="${env_file}" \
JETSON_REMOTE_ACCESS_PREFLIGHT="${sweep_access_preflight}" \
JETSON_REMOTE_GGUF_PREFLIGHT="${JETSON_REMOTE_GGUF_PREFLIGHT:-1}" \
JETSON_REMOTE_GGUF_PREFLIGHT_FAIL="${JETSON_REMOTE_GGUF_PREFLIGHT_FAIL:-0}" \
"${remote_sweep}" \
  --dry-run \
  --run-prefix "${run_prefix}" \
  --variant "${variant}"

printf 'remote_preflight_smoke=ok\n'
