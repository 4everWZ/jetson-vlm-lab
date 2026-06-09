#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_sweep="${JETSON_REMOTE_SWEEP:-${repo_root}/scripts/jetson/run_remote_optimization_sweep.sh}"
remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"

run_prefix="${JETSON_LIGHTWEIGHT_RUN_PREFIX:-lightweight-models-$(date -u +%Y%m%dT%H%M%SZ)}"
trial_count="${JETSON_LIGHTWEIGHT_TRIAL_COUNT:-5}"
max_tokens="${JETSON_LIGHTWEIGHT_MAX_TOKENS:-64}"
temperature="${JETSON_LIGHTWEIGHT_TEMPERATURE:-0}"
fake_stream_max_frames="${JETSON_LIGHTWEIGHT_FAKE_STREAM_MAX_FRAMES:-3}"
min_lfb_blocks="${JETSON_LIGHTWEIGHT_MIN_LFB_BLOCKS:-150}"
wait_timeout_s="${JETSON_LIGHTWEIGHT_WAIT_TIMEOUT_S:-600}"
remote_pythonpath="${JETSON_REMOTE_PYTHONPATH:-src}"
quality_review_policy="${JETSON_REMOTE_QUALITY_REVIEW_POLICY:-configs/benchmark/quality_review_policy.json}"
fail_on_promotion_precheck="${JETSON_LIGHTWEIGHT_FAIL_ON_PROMOTION_PRECHECK:-0}"
fail_on_startup_precheck="${JETSON_LIGHTWEIGHT_FAIL_ON_STARTUP_PRECHECK:-0}"
qwen3_selector="${JETSON_LIGHTWEIGHT_QWEN3_SELECTOR:-1}"
qwen3_fallback_min_lfb_blocks="${JETSON_LIGHTWEIGHT_QWEN3_FALLBACK_MIN_LFB_BLOCKS:-100}"

baseline_variants_text="${JETSON_LIGHTWEIGHT_BASELINE_VARIANTS:-minicpm-q4-baseline-b128-u32-kvq8 gemma-q4-baseline-gpu12-b512-u512-kvq8}"
candidate_variants_text="${JETSON_LIGHTWEIGHT_CANDIDATE_VARIANTS:-smolvlm2-256m-q8-smoke qwen3-vl-2b-thinking-q4-smoke youtu-vl-4b-q4-thirdparty-smoke}"
extra_variants_text="${JETSON_LIGHTWEIGHT_EXTRA_VARIANTS:-}"
manifest_path="${JETSON_LIGHTWEIGHT_MANIFEST:-outputs/optimization_sweeps/${run_prefix}/${run_prefix}.manifest.json}"
comparison_output="${JETSON_LIGHTWEIGHT_COMPARISON_OUTPUT:-outputs/optimization_sweeps/${run_prefix}/comparison.md}"

if [[ "${fail_on_promotion_precheck}" != "0" && "${fail_on_promotion_precheck}" != "1" ]]; then
  echo "JETSON_LIGHTWEIGHT_FAIL_ON_PROMOTION_PRECHECK must be 0 or 1." >&2
  exit 2
fi
if [[ "${fail_on_startup_precheck}" != "0" && "${fail_on_startup_precheck}" != "1" ]]; then
  echo "JETSON_LIGHTWEIGHT_FAIL_ON_STARTUP_PRECHECK must be 0 or 1." >&2
  exit 2
fi

read -r -a baseline_variants <<< "${baseline_variants_text}"
read -r -a candidate_variants <<< "${candidate_variants_text}"
read -r -a extra_variants <<< "${extra_variants_text}"

sweep_args=(
  --run-prefix "${run_prefix}"
  --trial-count "${trial_count}"
  --max-tokens "${max_tokens}"
  --temperature "${temperature}"
  --fake-stream-max-frames "${fake_stream_max_frames}"
  --min-lfb-blocks "${min_lfb_blocks}"
  --wait-timeout-s "${wait_timeout_s}"
)

for variant in "${baseline_variants[@]}" "${candidate_variants[@]}" "${extra_variants[@]}"; do
  if [[ -n "${variant}" ]]; then
    sweep_args+=(--variant "${variant}")
  fi
done

JETSON_REMOTE_PREPARE_MAX_CLOCKS=1 \
JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1 \
JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR="${qwen3_selector}" \
JETSON_REMOTE_QWEN3_INSTRUCT_FALLBACK_MIN_LFB_BLOCKS="${qwen3_fallback_min_lfb_blocks}" \
"${remote_sweep}" "${sweep_args[@]}"

"${remote_exec}" \
  "PYTHONPATH=${remote_pythonpath}" \
  python3 -m edge_vlm.sweep_quality_review \
  --manifest "${manifest_path}" \
  --policy "${quality_review_policy}" \
  --allow-failures

compare_args=(
  "PYTHONPATH=${remote_pythonpath}"
  python3
  -m
  edge_vlm.optimization
  compare
  --manifest "${manifest_path}"
)

for baseline in "${baseline_variants[@]}"; do
  if [[ -n "${baseline}" ]]; then
    compare_args+=(--baseline-variant "${baseline}")
  fi
done

compare_args+=(--startup-require-cached-artifacts)

if [[ "${fail_on_startup_precheck}" == "1" ]]; then
  compare_args+=(--fail-on-startup-precheck)
fi
if [[ "${fail_on_promotion_precheck}" == "1" ]]; then
  compare_args+=(--fail-on-promotion-precheck)
fi

compare_args+=(
  --ranking-min-lfb-blocks "${min_lfb_blocks}"
  --promotion-precheck-stage formal-repeat
  --promotion-require-startup-precheck
  --promotion-require-quality-review
  --output "${comparison_output}"
)

"${remote_exec}" "${compare_args[@]}"
