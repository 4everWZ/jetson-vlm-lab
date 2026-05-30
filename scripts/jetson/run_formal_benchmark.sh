#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${repo_root}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
run_id="${EDGE_VLM_FORMAL_RUN_ID:-jetson-formal-${timestamp}}"
config_path="${EDGE_VLM_CONFIG:-configs/models/minicpmv46_q4.yaml}"
cases_path="${EDGE_VLM_CASES:-configs/benchmark/prompt_cases.jsonl}"
output_path="${EDGE_VLM_OUTPUT:-outputs/benchmarks/${run_id}.jsonl}"
summary_path="${EDGE_VLM_SUMMARY_OUTPUT:-outputs/benchmarks/${run_id}.md}"
metadata_path="${EDGE_VLM_METADATA_OUTPUT:-outputs/benchmarks/${run_id}.manifest.json}"
profile_dir="${EDGE_VLM_PROFILE_DIR:-outputs/benchmarks/${run_id}.profile}"
trial_count="${EDGE_VLM_TRIAL_COUNT:-3}"
trial_delay_s="${EDGE_VLM_TRIAL_DELAY_S:-0}"
max_tokens="${EDGE_VLM_MAX_TOKENS:-64}"
temperature="${EDGE_VLM_TEMPERATURE:-0}"
python_bin="${PYTHON_BIN:-python3}"
dry_run="${EDGE_VLM_FORMAL_DRY_RUN:-0}"
skip_tegrastats="${EDGE_VLM_SKIP_TEGRASTATS:-0}"
tegrastats_interval_ms="${TEGR_STATS_INTERVAL_MS:-1000}"

mkdir -p "$(dirname "${output_path}")" "$(dirname "${summary_path}")" "$(dirname "${metadata_path}")" "${profile_dir}"

capture_command() {
  local name="$1"
  shift
  if command -v "$1" >/dev/null 2>&1; then
    "$@" > "${profile_dir}/${name}" 2>&1 || true
  else
    printf '%s not available\n' "$1" > "${profile_dir}/${name}"
  fi
}

capture_command uname.txt uname -a
capture_command docker-version.txt docker --version
capture_command nvpmodel.txt nvpmodel -q
capture_command jetson-clocks.txt jetson_clocks --show

export EDGE_VLM_POWER_MODE="${profile_dir}/nvpmodel.txt"
export EDGE_VLM_JETSON_CLOCKS="${profile_dir}/jetson-clocks.txt"

tegrastats_pid=""
cleanup() {
  if [[ -n "${tegrastats_pid}" ]] && kill -0 "${tegrastats_pid}" >/dev/null 2>&1; then
    kill "${tegrastats_pid}" >/dev/null 2>&1 || true
    wait "${tegrastats_pid}" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

if [[ "${skip_tegrastats}" == "1" ]]; then
  export EDGE_VLM_TEGRASTATS_STATUS="skipped"
  unset EDGE_VLM_TEGRASTATS_LOG
elif command -v tegrastats >/dev/null 2>&1; then
  tegrastats_log="${EDGE_VLM_TEGRASTATS_LOG:-outputs/tegrastats/${run_id}.log}"
  mkdir -p "$(dirname "${tegrastats_log}")"
  tegrastats --interval "${tegrastats_interval_ms}" > "${tegrastats_log}" 2>&1 &
  tegrastats_pid="$!"
  export EDGE_VLM_TEGRASTATS_LOG="${tegrastats_log}"
  export EDGE_VLM_TEGRASTATS_STATUS="running"
else
  export EDGE_VLM_TEGRASTATS_STATUS="not_available"
  unset EDGE_VLM_TEGRASTATS_LOG
fi

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

benchmark_args=(
  -m edge_vlm.benchmark
  --config "${config_path}"
  --cases "${cases_path}"
  --output "${output_path}"
  --summary-output "${summary_path}"
  --metadata-output "${metadata_path}"
  --run-id "${run_id}"
  --trial-count "${trial_count}"
  --trial-delay-s "${trial_delay_s}"
  --max-tokens "${max_tokens}"
  --temperature "${temperature}"
)

if [[ "${dry_run}" == "1" ]]; then
  benchmark_args+=(--dry-run)
fi

"${python_bin}" "${benchmark_args[@]}"

