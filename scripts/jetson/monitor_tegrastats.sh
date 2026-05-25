#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
interval_ms="${TEGR_STATS_INTERVAL_MS:-1000}"
log_dir="${TEGR_STATS_LOG_DIR:-${repo_root}/outputs/tegrastats}"
timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
log_file="${TEGR_STATS_LOG_FILE:-${log_dir}/tegrastats-${timestamp}.log}"

if ! command -v tegrastats >/dev/null 2>&1; then
  echo "tegrastats not found. This script is intended for NVIDIA Jetson devices." >&2
  exit 2
fi

mkdir -p "${log_dir}"
echo "Writing tegrastats to ${log_file}"
exec tegrastats --interval "${interval_ms}" | tee "${log_file}"
