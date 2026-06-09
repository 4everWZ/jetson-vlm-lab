#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_sweep="${JETSON_REMOTE_SWEEP:-${repo_root}/scripts/jetson/run_remote_optimization_sweep.sh}"
remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"

run_prefix="${JETSON_CURRENT_DEFAULTS_RUN_PREFIX:-current-defaults-$(date -u +%Y%m%dT%H%M%SZ)}"
trial_count="${JETSON_CURRENT_DEFAULTS_TRIAL_COUNT:-10}"
max_tokens="${JETSON_CURRENT_DEFAULTS_MAX_TOKENS:-64}"
temperature="${JETSON_CURRENT_DEFAULTS_TEMPERATURE:-0}"
fake_stream_max_frames="${JETSON_CURRENT_DEFAULTS_FAKE_STREAM_MAX_FRAMES:-3}"
min_lfb_blocks="${JETSON_CURRENT_DEFAULTS_MIN_LFB_BLOCKS:-150}"
wait_timeout_s="${JETSON_CURRENT_DEFAULTS_WAIT_TIMEOUT_S:-180}"
remote_pythonpath="${JETSON_REMOTE_PYTHONPATH:-src}"
quality_review_policy="${JETSON_REMOTE_QUALITY_REVIEW_POLICY:-configs/benchmark/quality_review_policy.json}"

minicpm_variant="${JETSON_CURRENT_DEFAULTS_MINICPM_VARIANT:-minicpm-q4-baseline-b128-u32-kvq8}"
gemma_variant="${JETSON_CURRENT_DEFAULTS_GEMMA_VARIANT:-gemma-q4-baseline-gpu12-b512-u512-kvq8}"
manifest_path="${JETSON_CURRENT_DEFAULTS_MANIFEST:-outputs/optimization_sweeps/${run_prefix}/${run_prefix}.manifest.json}"
comparison_output="${JETSON_CURRENT_DEFAULTS_COMPARISON_OUTPUT:-outputs/optimization_sweeps/${run_prefix}/comparison.md}"

JETSON_REMOTE_PREPARE_MAX_CLOCKS=1 \
JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1 \
"${remote_sweep}" \
  --run-prefix "${run_prefix}" \
  --variant "${minicpm_variant}" \
  --variant "${gemma_variant}" \
  --trial-count "${trial_count}" \
  --max-tokens "${max_tokens}" \
  --temperature "${temperature}" \
  --fake-stream-max-frames "${fake_stream_max_frames}" \
  --min-lfb-blocks "${min_lfb_blocks}" \
  --wait-timeout-s "${wait_timeout_s}"

"${remote_exec}" \
  "PYTHONPATH=${remote_pythonpath}" \
  python3 -m edge_vlm.sweep_quality_review \
  --manifest "${manifest_path}" \
  --policy "${quality_review_policy}" \
  --allow-failures

"${remote_exec}" \
  "PYTHONPATH=${remote_pythonpath}" \
  python3 -m edge_vlm.optimization compare \
  --manifest "${manifest_path}" \
  --baseline-variant "${minicpm_variant}" \
  --baseline-variant "${gemma_variant}" \
  --ranking-min-lfb-blocks "${min_lfb_blocks}" \
  --promotion-precheck-stage promotion-reference \
  --promotion-require-quality-review \
  --output "${comparison_output}"
