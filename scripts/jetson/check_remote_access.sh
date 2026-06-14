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
  if ! host_is_tailscale_cgnat_ipv4 "${candidate}"; then
    return 0
  fi
  if ! command -v tailscale >/dev/null 2>&1; then
    printf 'tailnet_probe=tailscale_cli_missing\n'
    return 0
  fi
  if tailscale status >/dev/null 2>&1; then
    printf 'tailnet_probe=tailscale_status_ok\n'
  else
    printf 'tailnet_probe=tailscale_status_failed\n'
  fi
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

if ! command -v nc >/dev/null 2>&1; then
  echo "tcp_probe=nc_missing" >&2
  echo "nc is required for the local TCP SSH precheck." >&2
  exit 2
fi

stderr_file="$(mktemp)"
cleanup() {
  rm -f "${stderr_file}"
}
trap cleanup EXIT

if nc -vz -w "${tcp_timeout}" "${host}" "${port}" 2>"${stderr_file}"; then
  printf 'tcp_probe=ok\n'
  exit 0
else
  status=$?
fi

echo "tcp_probe=tcp_connect_failed" >&2
cat "${stderr_file}" >&2
exit "${status}"
