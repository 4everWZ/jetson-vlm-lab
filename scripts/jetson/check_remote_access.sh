#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${JETSON_ENV_FILE:-${repo_root}/.env.jetson}"
. "${repo_root}/scripts/jetson/env_file.sh"
jetson_load_env_file "${env_file}"

host="${JETSON_SSH_HOST:-}"
user="${JETSON_SSH_USER:-}"
repo_dir="${JETSON_REPO_DIR:-~/code/jetson-vlm-lab}"
port="${JETSON_SSH_PORT:-22}"
dry_run="${JETSON_REMOTE_ACCESS_DRY_RUN:-0}"
tcp_timeout="${JETSON_REMOTE_ACCESS_TCP_TIMEOUT:-5}"
icmp_probe="${JETSON_REMOTE_ACCESS_ICMP_PROBE:-1}"
icmp_timeout="${JETSON_REMOTE_ACCESS_ICMP_TIMEOUT:-3}"
tcp_probe="${JETSON_REMOTE_ACCESS_TCP_PROBE:-auto}"
tailscale_bin="${JETSON_TAILSCALE_BIN:-}"
powershell_bin="${JETSON_POWERSHELL_BIN:-}"

if [[ -z "${host}" || -z "${user}" ]]; then
  echo "JETSON_SSH_HOST and JETSON_SSH_USER are required." >&2
  exit 2
fi

if [[ "${dry_run}" != "0" && "${dry_run}" != "1" ]]; then
  echo "JETSON_REMOTE_ACCESS_DRY_RUN must be 0 or 1." >&2
  exit 2
fi

if [[ "${icmp_probe}" != "0" && "${icmp_probe}" != "1" ]]; then
  echo "JETSON_REMOTE_ACCESS_ICMP_PROBE must be 0 or 1." >&2
  exit 2
fi

if [[ "${tcp_probe}" != "auto" && "${tcp_probe}" != "nc" && "${tcp_probe}" != "powershell" ]]; then
  echo "JETSON_REMOTE_ACCESS_TCP_PROBE must be auto, nc, or powershell." >&2
  exit 2
fi

resolve_tool() {
  local explicit="$1"
  shift
  local candidate
  if [[ -n "${explicit}" ]]; then
    if [[ -x "${explicit}" || -f "${explicit}" ]]; then
      printf '%s\n' "${explicit}"
      return 0
    fi
    return 1
  fi
  for candidate in "$@"; do
    if [[ "${candidate}" == */* ]]; then
      if [[ -x "${candidate}" || -f "${candidate}" ]]; then
        printf '%s\n' "${candidate}"
        return 0
      fi
    elif command -v "${candidate}" >/dev/null 2>&1; then
      command -v "${candidate}"
      return 0
    fi
  done
  return 1
}

host_is_tailscale_cgnat_ipv4() {
  local candidate="$1"
  local octet1 octet2 octet3 octet4 octet
  [[ "${candidate}" =~ ^[0-9]{1,3}(\.[0-9]{1,3}){3}$ ]] || return 1
  IFS=. read -r octet1 octet2 octet3 octet4 <<<"${candidate}"
  for octet in "${octet1}" "${octet2}" "${octet3}" "${octet4}"; do
    [[ "${octet}" =~ ^[0-9]+$ ]] || return 1
    ((octet >= 0 && octet <= 255)) || return 1
  done
  [[ "${octet1}" == "100" ]] || return 1
  ((octet2 >= 64 && octet2 <= 127))
}

emit_tailnet_probe() {
  local candidate="$1"
  local resolved_tailscale
  if ! host_is_tailscale_cgnat_ipv4 "${candidate}"; then
    return 0
  fi
  if ! resolved_tailscale="$(
    resolve_tool \
      "${tailscale_bin}" \
      tailscale \
      "/mnt/c/Program Files/Tailscale/tailscale.exe"
  )"; then
    printf 'tailnet_probe=tailscale_cli_missing\n'
    return 0
  fi
  if "${resolved_tailscale}" status >/dev/null 2>&1; then
    printf 'tailnet_probe=tailscale_status_ok\n'
  else
    printf 'tailnet_probe=tailscale_status_failed\n'
  fi
}

powershell_quote() {
  local value="$1"
  printf "'%s'" "${value//\'/\'\'}"
}

run_powershell_tcp_probe() {
  local candidate_host="$1"
  local candidate_port="$2"
  local resolved_powershell quoted_host
  if ! resolved_powershell="$(
    resolve_tool \
      "${powershell_bin}" \
      powershell.exe \
      pwsh \
      "/mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe"
  )"; then
    return 127
  fi
  quoted_host="$(powershell_quote "${candidate_host}")"
  "${resolved_powershell}" -NoProfile -Command \
    "if (Test-NetConnection -ComputerName ${quoted_host} -Port ${candidate_port} -InformationLevel Quiet) { exit 0 } else { exit 1 }"
}

printf 'ssh_target=%s@%s\n' "${user}" "${host}"
printf 'ssh_port=%s\n' "${port}"
printf 'repo_dir=%s\n' "${repo_dir}"

if [[ "${dry_run}" == "1" ]]; then
  printf 'icmp_probe=skipped_dry_run\n'
  printf 'tcp_probe=skipped_dry_run\n'
  exit 0
fi

if [[ "${icmp_probe}" == "0" ]]; then
  printf 'icmp_probe=disabled\n'
elif ! command -v ping >/dev/null 2>&1; then
  printf 'icmp_probe=ping_missing\n'
elif ping -c 1 -W "${icmp_timeout}" "${host}" >/dev/null 2>&1; then
  printf 'icmp_probe=ok\n'
else
  printf 'icmp_probe=failed\n'
fi

emit_tailnet_probe "${host}"

stderr_file="$(mktemp)"
cleanup() {
  rm -f "${stderr_file}"
}
trap cleanup EXIT

if [[ "${tcp_probe}" != "powershell" ]]; then
  if ! command -v nc >/dev/null 2>&1; then
    if [[ "${tcp_probe}" == "nc" ]]; then
      echo "tcp_probe=nc_missing" >&2
      echo "nc is required for the local TCP SSH precheck." >&2
      exit 2
    fi
  elif nc -vz -w "${tcp_timeout}" "${host}" "${port}" 2>"${stderr_file}"; then
    printf 'tcp_probe=ok\n'
    printf 'tcp_probe_transport=nc\n'
    exit 0
  else
    status=$?
    if [[ "${tcp_probe}" == "nc" ]]; then
      echo "tcp_probe=tcp_connect_failed" >&2
      cat "${stderr_file}" >&2
      exit "${status}"
    fi
  fi
fi

if [[ "${tcp_probe}" != "nc" ]]; then
  if run_powershell_tcp_probe "${host}" "${port}" 2>>"${stderr_file}"; then
    printf 'tcp_probe=ok\n'
    printf 'tcp_probe_transport=powershell\n'
    exit 0
  else
    status=$?
  fi
fi

if [[ "${status:-127}" -eq 127 && ! -s "${stderr_file}" ]]; then
  echo "tcp_probe=powershell_missing" >&2
  echo "nc or PowerShell is required for the local TCP SSH precheck." >&2
  exit 2
fi
echo "tcp_probe=tcp_connect_failed" >&2
cat "${stderr_file}" >&2
exit "${status:-1}"
