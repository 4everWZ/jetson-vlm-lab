#!/usr/bin/env bash

phase_now_ns() {
  date +%s%N
}

phase_duration_s() {
  local start_ns="$1"
  local end_ns="$2"
  local delta_ns=$((end_ns - start_ns))
  if ((delta_ns < 0)); then
    delta_ns=0
  fi
  local millis=$(((delta_ns + 500000) / 1000000))
  printf '%d.%03d' "$((millis / 1000))" "$((millis % 1000))"
}

write_launch_phase() {
  local phase="$1"
  local duration_s="$2"
  local status="$3"
  local phase_log="${EDGE_VLM_LAUNCH_PHASE_LOG:-}"
  if [[ -z "${phase_log}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${phase_log}")"
  printf '{"phase":"%s","available":true,"duration_s":%s,"reason":null,"source":"launcher","details":{"status":"%s"}}\n' \
    "${phase}" "${duration_s}" "${status}" >> "${phase_log}"
}

write_launch_phase_unavailable() {
  local phase="$1"
  local reason="$2"
  local status="$3"
  local phase_log="${EDGE_VLM_LAUNCH_PHASE_LOG:-}"
  if [[ -z "${phase_log}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${phase_log}")"
  printf '{"phase":"%s","available":false,"duration_s":null,"reason":"%s","source":"launcher","details":{"status":"%s"}}\n' \
    "${phase}" "${reason}" "${status}" >> "${phase_log}"
}
