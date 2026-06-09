#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_sweep="${JETSON_REMOTE_SWEEP:-${repo_root}/scripts/jetson/run_remote_optimization_sweep.sh}"
remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"

run_prefix="${JETSON_TENCENT_TEXT_RUN_PREFIX:-tencent-text-$(date -u +%Y%m%dT%H%M%SZ)}"
trial_count="${JETSON_TENCENT_TEXT_TRIAL_COUNT:-5}"
max_tokens="${JETSON_TENCENT_TEXT_MAX_TOKENS:-64}"
temperature="${JETSON_TENCENT_TEXT_TEMPERATURE:-0}"
fake_stream_max_frames="${JETSON_TENCENT_TEXT_FAKE_STREAM_MAX_FRAMES:-0}"
min_lfb_blocks="${JETSON_TENCENT_TEXT_MIN_LFB_BLOCKS:-150}"
wait_timeout_s="${JETSON_TENCENT_TEXT_WAIT_TIMEOUT_S:-600}"
remote_pythonpath="${JETSON_REMOTE_PYTHONPATH:-src}"
quality_review_policy="${JETSON_REMOTE_QUALITY_REVIEW_POLICY:-configs/benchmark/quality_review_policy.json}"
fail_on_promotion_precheck="${JETSON_TENCENT_TEXT_FAIL_ON_PROMOTION_PRECHECK:-0}"
fail_on_startup_precheck="${JETSON_TENCENT_TEXT_FAIL_ON_STARTUP_PRECHECK:-0}"

candidate_variants_text="${JETSON_TENCENT_TEXT_VARIANTS:-tencent-hy-mt1p5-1p8b-1p25bit-text-smoke tencent-hy-mt1p5-1p8b-2bit-text-smoke tencent-hy-mt1p5-1p8b-q4-text-smoke tencent-hy-mt1p5-1p8b-q6-text-smoke tencent-hy-mt1p5-1p8b-q8-text-smoke tencent-hy-mt2-1p8b-1p25bit-text-smoke tencent-hy-mt2-1p8b-2bit-text-smoke tencent-hy-mt2-1p8b-q4-text-smoke tencent-hy-mt2-1p8b-q6-text-smoke tencent-hy-mt2-1p8b-q8-text-smoke tencent-youtu-llm-2b-q8-text-smoke}"
extra_variants_text="${JETSON_TENCENT_TEXT_EXTRA_VARIANTS:-}"
manifest_path="${JETSON_TENCENT_TEXT_MANIFEST:-outputs/optimization_sweeps/${run_prefix}/${run_prefix}.manifest.json}"
comparison_output="${JETSON_TENCENT_TEXT_COMPARISON_OUTPUT:-outputs/optimization_sweeps/${run_prefix}/comparison.md}"

if [[ "${fail_on_promotion_precheck}" != "0" && "${fail_on_promotion_precheck}" != "1" ]]; then
  echo "JETSON_TENCENT_TEXT_FAIL_ON_PROMOTION_PRECHECK must be 0 or 1." >&2
  exit 2
fi
if [[ "${fail_on_startup_precheck}" != "0" && "${fail_on_startup_precheck}" != "1" ]]; then
  echo "JETSON_TENCENT_TEXT_FAIL_ON_STARTUP_PRECHECK must be 0 or 1." >&2
  exit 2
fi

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

for variant in "${candidate_variants[@]}" "${extra_variants[@]}"; do
  if [[ -n "${variant}" ]]; then
    sweep_args+=(--variant "${variant}")
  fi
done

JETSON_REMOTE_PREPARE_MAX_CLOCKS=1 \
JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT=1 \
"${remote_sweep}" "${sweep_args[@]}"

"${remote_exec}" \
  "PYTHONPATH=${remote_pythonpath}" \
  python3 \
  -m \
  edge_vlm.sweep_quality_review \
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
  --startup-require-cached-artifacts
  --ranking-min-lfb-blocks "${min_lfb_blocks}"
  --promotion-precheck-stage formal-repeat
  --promotion-require-startup-precheck
  --promotion-require-quality-review
  --output "${comparison_output}"
)

if [[ "${fail_on_startup_precheck}" == "1" ]]; then
  compare_args+=(--fail-on-startup-precheck)
fi
if [[ "${fail_on_promotion_precheck}" == "1" ]]; then
  compare_args+=(--fail-on-promotion-precheck)
fi

"${remote_exec}" "${compare_args[@]}"
