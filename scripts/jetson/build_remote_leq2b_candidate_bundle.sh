#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"

run_prefix="${JETSON_LEQ2B_BUNDLE_RUN_PREFIX:-leq2b-candidates-$(date -u +%Y%m%dT%H%M%SZ)}"
remote_pythonpath="${JETSON_REMOTE_PYTHONPATH:-src}"
selection_inputs_text="${JETSON_LEQ2B_BUNDLE_SELECTIONS:-}"
vlm_selection_dir="${JETSON_LEQ2B_VLM_SELECTION_DIR:-}"
text_selection_dir="${JETSON_LEQ2B_TEXT_SELECTION_DIR:-}"
bundle_output="${JETSON_LEQ2B_BUNDLE_OUTPUT:-outputs/optimization_sweeps/${run_prefix}/leq2b.candidate_bundle.json}"

selection_inputs=()
if [[ -n "${selection_inputs_text}" ]]; then
  read -r -a explicit_selection_inputs <<< "${selection_inputs_text}"
  for input_path in "${explicit_selection_inputs[@]}"; do
    if [[ -n "${input_path}" ]]; then
      selection_inputs+=("${input_path}")
    fi
  done
fi

if [[ -n "${vlm_selection_dir}" ]]; then
  selection_inputs+=(
    "${vlm_selection_dir}/ranking.leq2b-vlm.selection.json"
    "${vlm_selection_dir}/promotion.leq2b-vlm.selection.json"
  )
fi

if [[ -n "${text_selection_dir}" ]]; then
  selection_inputs+=(
    "${text_selection_dir}/ranking.leq2b-text.selection.json"
    "${text_selection_dir}/promotion.leq2b-text.selection.json"
  )
fi

if [[ "${#selection_inputs[@]}" -eq 0 ]]; then
  echo "Set JETSON_LEQ2B_BUNDLE_SELECTIONS or at least one lane selection directory before building the bundle." >&2
  exit 2
fi

bundle_args=(
  "PYTHONPATH=${remote_pythonpath}"
  python3
  -m
  edge_vlm.optimization
  bundle-selections
)

for input_path in "${selection_inputs[@]}"; do
  bundle_args+=(--input "${input_path}")
done
bundle_args+=(--output "${bundle_output}")

"${remote_exec}" "${bundle_args[@]}"
