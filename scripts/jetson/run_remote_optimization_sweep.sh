#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${JETSON_ENV_FILE:-${repo_root}/.env.jetson}"
. "${repo_root}/scripts/jetson/env_file.sh"
jetson_load_env_file "${env_file}"

remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"
remote_sync="${JETSON_REMOTE_SYNC:-1}"
remote_branch="${JETSON_REMOTE_BRANCH:-main}"
remote_pythonpath="${JETSON_REMOTE_PYTHONPATH:-src}"
llama_cpp_image="${JETSON_REMOTE_LLAMA_CPP_IMAGE:-ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87}"
access_preflight="${JETSON_REMOTE_ACCESS_PREFLIGHT:-0}"
prepare_max_clocks="${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-0}"
drop_caches_before_variant="${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-0}"
memory_prepare_attempts="${JETSON_REMOTE_MEMORY_PREPARE_ATTEMPTS:-1}"
qwen3_selector="${JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR:-0}"
qwen3_primary_variant="${JETSON_REMOTE_QWEN3_INSTRUCT_PRIMARY_VARIANT:-qwen3-vl-2b-instruct-q4-smoke}"
qwen3_fallback_variant="${JETSON_REMOTE_QWEN3_INSTRUCT_FALLBACK_VARIANT:-qwen3-vl-2b-instruct-q8-smoke}"
qwen3_fallback_min_lfb_blocks="${JETSON_REMOTE_QWEN3_INSTRUCT_FALLBACK_MIN_LFB_BLOCKS:-}"
qwen3_selector_output="${JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR_OUTPUT:-}"
qwen3_memory_diagnostics="${JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS:-1}"
qwen3_memory_diagnostics_sudo="${JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO:-0}"
qwen3_cma_experiment_plan="${JETSON_REMOTE_QWEN3_INSTRUCT_CMA_PLAN:-1}"
qwen3_cma_apply_dry_run="${JETSON_REMOTE_QWEN3_INSTRUCT_CMA_APPLY_DRY_RUN:-1}"
qwen3_cma_apply_candidate_id="${JETSON_REMOTE_QWEN3_INSTRUCT_CMA_APPLY_CANDIDATE_ID:-max-required-lfb-rounded-64mib}"
gguf_preflight="${JETSON_REMOTE_GGUF_PREFLIGHT:-0}"
gguf_preflight_fail="${JETSON_REMOTE_GGUF_PREFLIGHT_FAIL:-0}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <edge_vlm.jetson_sweep args...>" >&2
  exit 2
fi

if [[ "${access_preflight}" == "1" ]]; then
  JETSON_ENV_FILE="${env_file}" "${repo_root}/scripts/jetson/check_remote_access.sh"
elif [[ "${access_preflight}" != "0" ]]; then
  echo "JETSON_REMOTE_ACCESS_PREFLIGHT must be 0 or 1." >&2
  exit 2
fi

if [[ "${remote_sync}" == "1" ]]; then
  "${remote_exec}" git fetch origin "${remote_branch}"
  "${remote_exec}" git checkout --detach FETCH_HEAD
elif [[ "${remote_sync}" != "0" ]]; then
  echo "JETSON_REMOTE_SYNC must be 0 or 1." >&2
  exit 2
fi

sudo_password_from_env() {
  printf '%s' "${JETSON_REMOTE_SUDO_PASSWORD:-${JETSON_SSH_PASSWORD:-}}"
}

extract_arg_value() {
  local flag="$1"
  shift
  local previous=""
  for arg in "$@"; do
    if [[ "${previous}" == "${flag}" ]]; then
      printf '%s' "${arg}"
      return 0
    fi
    if [[ "${arg}" == "${flag}="* ]]; then
      printf '%s' "${arg#*=}"
      return 0
    fi
    previous="${arg}"
  done
  return 1
}

has_variant_arg() {
  local wanted="$1"
  shift
  local previous=""
  for arg in "$@"; do
    if [[ "${previous}" == "--variant" && "${arg}" == "${wanted}" ]]; then
      return 0
    fi
    if [[ "${arg}" == "--variant=${wanted}" ]]; then
      return 0
    fi
    previous="${arg}"
  done
  return 1
}

has_variant_min_lfb_override_arg() {
  local wanted_variant="$1"
  shift
  local previous=""
  for arg in "$@"; do
    if [[ "${previous}" == "--variant-min-lfb-blocks" && "${arg%%=*}" == "${wanted_variant}" ]]; then
      return 0
    fi
    if [[ "${arg}" == --variant-min-lfb-blocks="${wanted_variant}="* ]]; then
      return 0
    fi
    previous="${arg}"
  done
  return 1
}

memory_prepare_shell() {
  local attempts="$1"
  printf 'prepare_attempt=1; while [ "${prepare_attempt}" -le %s ]; do sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; prepare_attempt=$((prepare_attempt + 1)); done' "${attempts}"
}

run_remote_memory_prepare() {
  local prepare_command
  prepare_command="$(memory_prepare_shell "${memory_prepare_attempts}")"
  printf '%s\n' "${sudo_password}" | "${remote_exec}" \
    sudo -S -p '' sh -c "${prepare_command}"
}

derive_selector_memory_diagnostics_output() {
  local selector_output="$1"
  if [[ "${selector_output}" == *.json ]]; then
    printf '%s.memory-diagnostics.json' "${selector_output%.json}"
  else
    printf '%s.memory-diagnostics.json' "${selector_output}"
  fi
}

derive_sudo_memory_diagnostics_output() {
  local diagnostics_output="$1"
  if [[ "${diagnostics_output}" == *.json ]]; then
    printf '%s.sudo.json' "${diagnostics_output%.json}"
  else
    printf '%s.sudo.json' "${diagnostics_output}"
  fi
}

derive_cma_experiment_plan_output() {
  local selector_output="$1"
  if [[ "${selector_output}" == *.json ]]; then
    printf '%s.cma-experiment-plan.json' "${selector_output%.json}"
  else
    printf '%s.cma-experiment-plan.json' "${selector_output}"
  fi
}

derive_boot_config_cma_plan_output() {
  local selector_output="$1"
  if [[ "${selector_output}" == *.json ]]; then
    printf '%s.boot-config-cma-plan.json' "${selector_output%.json}"
  else
    printf '%s.boot-config-cma-plan.json' "${selector_output}"
  fi
}

derive_boot_config_cma_apply_dry_run_output() {
  local selector_output="$1"
  if [[ "${selector_output}" == *.json ]]; then
    printf '%s.boot-config-cma-apply-dry-run.json' "${selector_output%.json}"
  else
    printf '%s.boot-config-cma-apply-dry-run.json' "${selector_output}"
  fi
}

run_remote_sudo_memory_diagnostics() {
  local output="$1"
  printf '%s\n' "${sudo_password}" | "${remote_exec}" \
    sudo -S -p '' \
    env "PYTHONPATH=${remote_pythonpath}" \
    python3 -m edge_vlm.jetson_memory_diagnostics \
    --output "${output}"
}

enrich_remote_selector_with_sudo_diagnostics() {
  local selector_output="$1"
  local sudo_output="$2"
  "${remote_exec}" \
    env "PYTHONPATH=${remote_pythonpath}" \
    python3 -m edge_vlm.selection_context_diagnostics \
    --selector-output "${selector_output}" \
    --sudo-memory-diagnostics-output "${sudo_output}"
}

write_remote_cma_experiment_plan() {
  local selector_output="$1"
  local plan_output="$2"
  "${remote_exec}" \
    env "PYTHONPATH=${remote_pythonpath}" \
    python3 -m edge_vlm.cma_experiment_plan \
    --selector-json "${selector_output}" \
    --output "${plan_output}"
}

write_remote_boot_config_cma_plan() {
  local cma_plan_output="$1"
  local boot_config_plan_output="$2"
  "${remote_exec}" \
    env "PYTHONPATH=${remote_pythonpath}" \
    python3 -m edge_vlm.boot_config_cma_plan \
    --cma-plan "${cma_plan_output}" \
    --output "${boot_config_plan_output}"
}

write_remote_boot_config_cma_apply_dry_run() {
  local boot_config_plan_output="$1"
  local candidate_id="$2"
  local apply_output="$3"
  "${remote_exec}" \
    env "PYTHONPATH=${remote_pythonpath}" \
    python3 -m edge_vlm.boot_config_cma_apply \
    --plan "${boot_config_plan_output}" \
    --candidate-id "${candidate_id}" \
    --output "${apply_output}"
}

if [[ "${prepare_max_clocks}" == "1" ]]; then
  sudo_password="$(sudo_password_from_env)"
  if [[ -z "${sudo_password}" ]]; then
    echo "JETSON_REMOTE_PREPARE_MAX_CLOCKS requires JETSON_REMOTE_SUDO_PASSWORD or JETSON_SSH_PASSWORD." >&2
    exit 2
  fi
  clocks_capture="outputs/jetson_inspect/jetson-clocks-max-$(date -u +%Y%m%dT%H%M%SZ).txt"
  printf '%s\n' "${sudo_password}" | "${remote_exec}" \
    sudo -S sh -c "mkdir -p outputs/jetson_inspect && jetson_clocks && jetson_clocks --show > ${clocks_capture}"
elif [[ "${prepare_max_clocks}" != "0" ]]; then
  echo "JETSON_REMOTE_PREPARE_MAX_CLOCKS must be 0 or 1." >&2
  exit 2
fi

has_pre_variant_command=0
for arg in "$@"; do
  if [[ "${arg}" == "--pre-variant-command" || "${arg}" == --pre-variant-command=* ]]; then
    has_pre_variant_command=1
    break
  fi
done

sweep_args=("$@")
if [[ "${drop_caches_before_variant}" == "1" ]]; then
  sudo_password="$(sudo_password_from_env)"
  if [[ -z "${sudo_password}" ]]; then
    echo "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT requires JETSON_REMOTE_SUDO_PASSWORD or JETSON_SSH_PASSWORD." >&2
    exit 2
  fi
  if ! [[ "${memory_prepare_attempts}" =~ ^[1-9][0-9]*$ ]]; then
    echo "JETSON_REMOTE_MEMORY_PREPARE_ATTEMPTS must be a positive integer." >&2
    exit 2
  fi
  if [[ "${has_pre_variant_command}" == "1" ]]; then
    echo "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT cannot be combined with --pre-variant-command." >&2
    exit 2
  fi
elif [[ "${drop_caches_before_variant}" != "0" ]]; then
  echo "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT must be 0 or 1." >&2
  exit 2
fi

if [[ "${qwen3_selector}" == "1" ]]; then
  if [[ "${qwen3_memory_diagnostics}" != "0" && "${qwen3_memory_diagnostics}" != "1" ]]; then
    echo "JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS must be 0 or 1." >&2
    exit 2
  fi
  if [[ "${qwen3_memory_diagnostics_sudo}" != "0" && "${qwen3_memory_diagnostics_sudo}" != "1" ]]; then
    echo "JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO must be 0 or 1." >&2
    exit 2
  fi
  if [[ "${qwen3_cma_experiment_plan}" != "0" && "${qwen3_cma_experiment_plan}" != "1" ]]; then
    echo "JETSON_REMOTE_QWEN3_INSTRUCT_CMA_PLAN must be 0 or 1." >&2
    exit 2
  fi
  if [[ "${qwen3_cma_apply_dry_run}" != "0" && "${qwen3_cma_apply_dry_run}" != "1" ]]; then
    echo "JETSON_REMOTE_QWEN3_INSTRUCT_CMA_APPLY_DRY_RUN must be 0 or 1." >&2
    exit 2
  fi
  if [[ "${qwen3_cma_apply_dry_run}" == "1" && -z "${qwen3_cma_apply_candidate_id}" ]]; then
    echo "JETSON_REMOTE_QWEN3_INSTRUCT_CMA_APPLY_CANDIDATE_ID must not be empty." >&2
    exit 2
  fi
  selector_min_lfb_blocks="$(extract_arg_value --min-lfb-blocks "${sweep_args[@]}" || true)"
  selector_run_prefix="$(extract_arg_value --run-prefix "${sweep_args[@]}" || true)"
  if [[ -z "${selector_min_lfb_blocks}" ]]; then
    selector_min_lfb_blocks="150"
  fi
  if [[ -z "${qwen3_selector_output}" && -n "${selector_run_prefix}" ]]; then
    qwen3_selector_output="outputs/optimization_sweeps/${selector_run_prefix}/${selector_run_prefix}.qwen3-selector.json"
  fi
  selector_memory_diagnostics_output=""
  if [[ -n "${qwen3_selector_output}" ]]; then
    selector_memory_diagnostics_output="$(derive_selector_memory_diagnostics_output "${qwen3_selector_output}")"
  fi
  if [[ "${qwen3_memory_diagnostics_sudo}" == "1" ]]; then
    sudo_password="$(sudo_password_from_env)"
    if [[ -z "${sudo_password}" ]]; then
      echo "JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO requires JETSON_REMOTE_SUDO_PASSWORD or JETSON_SSH_PASSWORD." >&2
      exit 2
    fi
    if [[ -z "${selector_memory_diagnostics_output}" ]]; then
      echo "JETSON_REMOTE_QWEN3_INSTRUCT_MEMORY_DIAGNOSTICS_SUDO requires --run-prefix or JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR_OUTPUT." >&2
      exit 2
    fi
  fi
  if [[ "${drop_caches_before_variant}" == "1" ]]; then
    run_remote_memory_prepare
  fi
  selector_args=(
    env
    "LLAMA_CPP_DOCKER_IMAGE=${llama_cpp_image}"
    "PYTHONPATH=${remote_pythonpath}"
    bash
    scripts/jetson/select_qwen3_instruct_variant.sh
    --primary-variant "${qwen3_primary_variant}"
    --fallback-variant "${qwen3_fallback_variant}"
    --min-lfb-blocks "${selector_min_lfb_blocks}"
  )
  if [[ -n "${qwen3_fallback_min_lfb_blocks}" ]]; then
    selector_args+=(--fallback-min-lfb-blocks "${qwen3_fallback_min_lfb_blocks}")
  fi
  if [[ -n "${qwen3_selector_output}" ]]; then
    selector_args+=(--output "${qwen3_selector_output}")
  fi
  if [[ "${qwen3_memory_diagnostics}" == "1" && -n "${selector_memory_diagnostics_output}" ]]; then
    selector_args+=(--memory-diagnostics-output "${selector_memory_diagnostics_output}")
  fi
  selector_json="$("${remote_exec}" "${selector_args[@]}")"
  selected_variant_id="$(printf '%s' "${selector_json}" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("selected_variant_id") or "")')"
  selected_reason="$(printf '%s' "${selector_json}" | python3 -c 'import json, sys; print(json.load(sys.stdin).get("selected_reason") or "")')"
  if [[ -n "${qwen3_selector_output}" ]]; then
    sweep_args+=(--selection-context-json "${qwen3_selector_output}")
  fi
  if [[ -n "${selected_variant_id}" && "${selected_variant_id}" == "${qwen3_fallback_variant}" && -n "${qwen3_fallback_min_lfb_blocks}" ]]; then
    if ! has_variant_min_lfb_override_arg "${selected_variant_id}" "${sweep_args[@]}"; then
      sweep_args+=(--variant-min-lfb-blocks "${selected_variant_id}=${qwen3_fallback_min_lfb_blocks}")
    fi
  fi
  if [[ -n "${selected_variant_id}" ]]; then
    if ! has_variant_arg "${selected_variant_id}" "${sweep_args[@]}"; then
      sweep_args+=(--variant "${selected_variant_id}")
    fi
  fi
  if [[ "${qwen3_memory_diagnostics_sudo}" == "1" ]]; then
    sudo_memory_diagnostics_output="$(derive_sudo_memory_diagnostics_output "${selector_memory_diagnostics_output}")"
    sudo_memory_diagnostics_json="$(run_remote_sudo_memory_diagnostics "${sudo_memory_diagnostics_output}")"
    sudo_memory_diagnostics_context_json="$(enrich_remote_selector_with_sudo_diagnostics "${qwen3_selector_output}" "${sudo_memory_diagnostics_output}")"
    echo "Qwen3 selector sudo memory diagnostics output: ${sudo_memory_diagnostics_output}" >&2
  fi
  if [[ "${qwen3_cma_experiment_plan}" == "1" && -z "${selected_variant_id}" && -n "${qwen3_selector_output}" ]]; then
    selector_cma_experiment_plan_output="$(derive_cma_experiment_plan_output "${qwen3_selector_output}")"
    selector_cma_experiment_plan_json="$(write_remote_cma_experiment_plan "${qwen3_selector_output}" "${selector_cma_experiment_plan_output}")"
    echo "Qwen3 selector CMA experiment plan output: ${selector_cma_experiment_plan_output}" >&2
    selector_boot_config_cma_plan_output="$(derive_boot_config_cma_plan_output "${qwen3_selector_output}")"
    selector_boot_config_cma_plan_json="$(write_remote_boot_config_cma_plan "${selector_cma_experiment_plan_output}" "${selector_boot_config_cma_plan_output}")"
    echo "Qwen3 selector boot config CMA patch plan output: ${selector_boot_config_cma_plan_output}" >&2
    if [[ "${qwen3_cma_apply_dry_run}" == "1" ]]; then
      selector_boot_config_cma_apply_output="$(derive_boot_config_cma_apply_dry_run_output "${qwen3_selector_output}")"
      selector_boot_config_cma_apply_json="$(write_remote_boot_config_cma_apply_dry_run "${selector_boot_config_cma_plan_output}" "${qwen3_cma_apply_candidate_id}" "${selector_boot_config_cma_apply_output}")"
      echo "Qwen3 selector boot config CMA apply dry-run output: ${selector_boot_config_cma_apply_output}" >&2
    fi
  fi
  if [[ -n "${selected_variant_id}" ]]; then
    echo "Qwen3 selector chose ${selected_variant_id} (${selected_reason:-unknown_reason})." >&2
  else
    echo "Qwen3 selector chose no variant (${selected_reason:-unknown_reason}); continuing without an auto-selected Qwen3 lane." >&2
  fi
elif [[ "${qwen3_selector}" != "0" ]]; then
  echo "JETSON_REMOTE_QWEN3_INSTRUCT_SELECTOR must be 0 or 1." >&2
  exit 2
fi

if [[ "${gguf_preflight}" == "1" ]]; then
  if [[ "${gguf_preflight_fail}" != "0" && "${gguf_preflight_fail}" != "1" ]]; then
    echo "JETSON_REMOTE_GGUF_PREFLIGHT_FAIL must be 0 or 1." >&2
    exit 2
  fi
  preflight_run_prefix="$(extract_arg_value --run-prefix "${sweep_args[@]}" || true)"
  if [[ -z "${preflight_run_prefix}" ]]; then
    echo "JETSON_REMOTE_GGUF_PREFLIGHT requires an explicit --run-prefix." >&2
    exit 2
  fi
  preflight_dir="outputs/optimization_sweeps/${preflight_run_prefix}"
  preflight_plan="${preflight_dir}/${preflight_run_prefix}.preflight-plan.json"
  preflight_manifest="${preflight_dir}/${preflight_run_prefix}.gguf-artifacts.json"
  preflight_prepare_env=()
  if [[ "${prepare_max_clocks}" == "1" ]]; then
    preflight_prepare_env+=(
      "EDGE_VLM_PREPARE_MAX_CLOCKS_ENABLED=${prepare_max_clocks}"
      "EDGE_VLM_PREPARE_MAX_CLOCKS_CAPTURE=${clocks_capture:-}"
    )
  fi
  if [[ "${drop_caches_before_variant}" == "1" ]]; then
    preflight_prepare_env+=(
      "EDGE_VLM_DROP_CACHES_BEFORE_VARIANT=${drop_caches_before_variant}"
      "EDGE_VLM_MEMORY_PREPARE_ATTEMPTS=${memory_prepare_attempts}"
      "EDGE_VLM_PRE_VARIANT_COMMAND_SOURCE=remote_wrapper_drop_caches"
    )
  fi
  "${remote_exec}" \
    env \
    "LLAMA_CPP_DOCKER_IMAGE=${llama_cpp_image}" \
    "PYTHONPATH=${remote_pythonpath}" \
    "${preflight_prepare_env[@]}" \
    python3 -m edge_vlm.jetson_sweep \
    "${sweep_args[@]}" \
    --dry-run \
    --plan-output "${preflight_plan}"
  preflight_exit=0
  if "${remote_exec}" \
    env \
    "PYTHONPATH=${remote_pythonpath}" \
    python3 -m edge_vlm.gguf_artifacts \
    check-plan \
    --plan "${preflight_plan}" \
    --output "${preflight_manifest}"; then
    preflight_exit=0
  else
    preflight_exit=$?
  fi
  if [[ "${preflight_exit}" -ne 0 ]]; then
    if [[ "${gguf_preflight_fail}" == "1" ]]; then
      echo "GGUF artifact preflight failed; aborting because JETSON_REMOTE_GGUF_PREFLIGHT_FAIL=1." >&2
      exit "${preflight_exit}"
    fi
    echo "GGUF artifact preflight failed; continuing because JETSON_REMOTE_GGUF_PREFLIGHT_FAIL=0. See ${preflight_manifest}." >&2
  fi
elif [[ "${gguf_preflight}" != "0" ]]; then
  echo "JETSON_REMOTE_GGUF_PREFLIGHT must be 0 or 1." >&2
  exit 2
fi

if [[ "${drop_caches_before_variant}" == "1" ]]; then
  printf '%s\n' "${sudo_password}" | "${remote_exec}" \
    bash -lc '
set -Eeuo pipefail
IFS= read -r sudo_password
memory_prepare_attempts="${1:?}"
shift
pw_fifo="$(mktemp -u "${TMPDIR:-/tmp}/edge-vlm-sudo.XXXXXX")"
mkfifo "${pw_fifo}"
chmod 600 "${pw_fifo}"
pw_feeder_pid=""
cleanup() {
  if [[ -n "${pw_feeder_pid}" ]]; then
    kill "${pw_feeder_pid}" >/dev/null 2>&1 || true
  fi
  rm -f "${pw_fifo}"
}
trap cleanup EXIT
(
  trap '\'''\'' PIPE
  while true; do
    if ! printf "%s\n" "${sudo_password}" > "${pw_fifo}" 2>/dev/null; then
      sleep 0.1
    fi
  done
) &
pw_feeder_pid="$!"
printf -v prepare_command "prepare_attempt=1; while [ \"\${prepare_attempt}\" -le %s ]; do sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory; prepare_attempt=\$((prepare_attempt + 1)); done" "${memory_prepare_attempts}"
pre_variant_command="sudo -S -p '\'''\'' sh -c '\''${prepare_command}'\'' < ${pw_fifo}"
"$@" --pre-variant-command "${pre_variant_command}"
' \
    remote-sweep \
    "${memory_prepare_attempts}" \
    env \
    "LLAMA_CPP_DOCKER_IMAGE=${llama_cpp_image}" \
    "PYTHONPATH=${remote_pythonpath}" \
    "EDGE_VLM_PREPARE_MAX_CLOCKS_ENABLED=${prepare_max_clocks}" \
    "EDGE_VLM_PREPARE_MAX_CLOCKS_CAPTURE=${clocks_capture:-}" \
    "EDGE_VLM_DROP_CACHES_BEFORE_VARIANT=${drop_caches_before_variant}" \
    "EDGE_VLM_MEMORY_PREPARE_ATTEMPTS=${memory_prepare_attempts}" \
    "EDGE_VLM_PRE_VARIANT_COMMAND_SOURCE=remote_wrapper_drop_caches" \
    bash scripts/jetson/run_optimization_sweep.sh \
    "${sweep_args[@]}"
  exit $?
fi

exec "${remote_exec}" \
  env \
  "LLAMA_CPP_DOCKER_IMAGE=${llama_cpp_image}" \
  "PYTHONPATH=${remote_pythonpath}" \
  "EDGE_VLM_PREPARE_MAX_CLOCKS_ENABLED=${prepare_max_clocks}" \
  "EDGE_VLM_PREPARE_MAX_CLOCKS_CAPTURE=${clocks_capture:-}" \
  "EDGE_VLM_DROP_CACHES_BEFORE_VARIANT=${drop_caches_before_variant}" \
  "EDGE_VLM_PRE_VARIANT_COMMAND_SOURCE=" \
  bash scripts/jetson/run_optimization_sweep.sh \
  "${sweep_args[@]}"
