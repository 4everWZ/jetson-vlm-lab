#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"
remote_sync="${JETSON_REMOTE_SYNC:-1}"
remote_pythonpath="${JETSON_REMOTE_PYTHONPATH:-src}"
llama_cpp_image="${JETSON_REMOTE_LLAMA_CPP_IMAGE:-ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87}"
prepare_max_clocks="${JETSON_REMOTE_PREPARE_MAX_CLOCKS:-0}"
drop_caches_before_variant="${JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT:-0}"

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <edge_vlm.jetson_sweep args...>" >&2
  exit 2
fi

if [[ "${remote_sync}" == "1" ]]; then
  "${remote_exec}" git pull --ff-only
elif [[ "${remote_sync}" != "0" ]]; then
  echo "JETSON_REMOTE_SYNC must be 0 or 1." >&2
  exit 2
fi

sudo_password_from_env() {
  printf '%s' "${JETSON_REMOTE_SUDO_PASSWORD:-${JETSON_SSH_PASSWORD:-}}"
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
  if [[ "${has_pre_variant_command}" == "1" ]]; then
    echo "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT cannot be combined with --pre-variant-command." >&2
    exit 2
  fi
elif [[ "${drop_caches_before_variant}" != "0" ]]; then
  echo "JETSON_REMOTE_DROP_CACHES_BEFORE_VARIANT must be 0 or 1." >&2
  exit 2
fi

if [[ "${drop_caches_before_variant}" == "1" ]]; then
  printf '%s\n' "${sudo_password}" | "${remote_exec}" \
    bash -lc '
set -Eeuo pipefail
IFS= read -r sudo_password
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
pre_variant_command="sudo -S -p '\'''\'' sh -c '\''sync; echo 3 > /proc/sys/vm/drop_caches'\'' < ${pw_fifo}"
"$@" --pre-variant-command "${pre_variant_command}"
' \
    remote-sweep \
    env \
    "LLAMA_CPP_DOCKER_IMAGE=${llama_cpp_image}" \
    "PYTHONPATH=${remote_pythonpath}" \
    bash scripts/jetson/run_optimization_sweep.sh \
    "${sweep_args[@]}"
  exit $?
fi

exec "${remote_exec}" \
  env \
  "LLAMA_CPP_DOCKER_IMAGE=${llama_cpp_image}" \
  "PYTHONPATH=${remote_pythonpath}" \
  bash scripts/jetson/run_optimization_sweep.sh \
  "${sweep_args[@]}"
