#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
remote_exec="${JETSON_REMOTE_EXEC:-${repo_root}/scripts/jetson/remote_exec.sh}"
dry_run="${JETSON_REMOTE_PROBE_DRY_RUN:-0}"
marker="${JETSON_REMOTE_PROBE_MARKER:-edge-vlm-remote-probe}"

probe_args=(printf '%s\n' "${marker}")

if [[ "${dry_run}" == "1" ]]; then
  JETSON_REMOTE_DRY_RUN=1 "${remote_exec}" "${probe_args[@]}"
  exit $?
elif [[ "${dry_run}" != "0" ]]; then
  echo "JETSON_REMOTE_PROBE_DRY_RUN must be 0 or 1." >&2
  exit 2
fi

stdout_file="$(mktemp)"
stderr_file="$(mktemp)"
cleanup() {
  rm -f "${stdout_file}" "${stderr_file}"
}
trap cleanup EXIT

status=0
"${remote_exec}" "${probe_args[@]}" >"${stdout_file}" 2>"${stderr_file}" || status=$?

stdout_text="$(cat "${stdout_file}")"
stderr_text="$(cat "${stderr_file}")"

if [[ "${status}" -eq 0 && "${stdout_text}" == *"${marker}"* ]]; then
  printf 'remote_probe=ok\n'
  exit 0
fi

classification="remote_command_failed"
if [[ "${stderr_text}" == *"Connection timed out"* ]]; then
  classification="ssh_connect_timeout"
elif [[ "${stderr_text}" == *"No route to host"* || "${stderr_text}" == *"Network is unreachable"* ]]; then
  classification="ssh_network_unreachable"
elif [[ "${stderr_text}" == *"Permission denied"* ]]; then
  classification="ssh_auth_failed"
elif [[ "${status}" -eq 255 ]]; then
  classification="ssh_failed"
fi

printf 'remote_probe=%s\n' "${classification}" >&2
if [[ -n "${stderr_text}" ]]; then
  printf '%s\n' "${stderr_text}" >&2
fi
if [[ -n "${stdout_text}" ]]; then
  printf '%s\n' "${stdout_text}"
fi
exit "${status}"
