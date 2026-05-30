#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
env_file="${JETSON_ENV_FILE:-${repo_root}/.env.jetson}"

trim() {
  local value="$1"
  value="${value#"${value%%[![:space:]]*}"}"
  value="${value%"${value##*[![:space:]]}"}"
  printf '%s' "${value}"
}

load_env_file() {
  local line key value
  while IFS= read -r line || [[ -n "${line}" ]]; do
    line="$(trim "${line}")"
    [[ -z "${line}" || "${line}" == \#* ]] && continue
    if [[ "${line}" == export\ * ]]; then
      line="$(trim "${line#export }")"
    fi
    [[ "${line}" == *=* ]] || continue
    key="$(trim "${line%%=*}")"
    value="$(trim "${line#*=}")"
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "${value}" == \"*\" && "${value}" == *\" ]]; then
      value="${value:1:${#value}-2}"
    elif [[ "${value}" == \'*\' && "${value}" == *\' ]]; then
      value="${value:1:${#value}-2}"
    fi
    export "${key}=${value}"
  done < "$1"
}

if [[ -f "${env_file}" ]]; then
  load_env_file "${env_file}"
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: $0 <remote command> [args...]" >&2
  exit 2
fi

host="${JETSON_SSH_HOST:-}"
user="${JETSON_SSH_USER:-}"
repo_dir="${JETSON_REPO_DIR:-~/code/jetson-vlm-lab}"
port="${JETSON_SSH_PORT:-22}"
dry_run="${JETSON_REMOTE_DRY_RUN:-0}"
strict_host_key_checking="${JETSON_SSH_STRICT_HOST_KEY_CHECKING:-accept-new}"

if [[ -z "${host}" || -z "${user}" ]]; then
  echo "JETSON_SSH_HOST and JETSON_SSH_USER are required." >&2
  exit 2
fi

quote_remote_path() {
  case "$1" in
    "~" | "~/"*) printf '%s' "$1" ;;
    *) printf '%q' "$1" ;;
  esac
}

remote_args=()
for arg in "$@"; do
  printf -v quoted_arg '%q' "${arg}"
  remote_args+=("${quoted_arg}")
done

target="${user}@${host}"
remote_repo_dir="$(quote_remote_path "${repo_dir}")"
remote_command="cd ${remote_repo_dir} && ${remote_args[*]}"

ssh_cmd=(ssh -p "${port}" -o "StrictHostKeyChecking=${strict_host_key_checking}")
if [[ -n "${JETSON_SSH_OPTS:-}" ]]; then
  read -r -a extra_ssh_opts <<< "${JETSON_SSH_OPTS}"
  ssh_cmd+=("${extra_ssh_opts[@]}")
fi
ssh_cmd+=("${target}" "${remote_command}")

if [[ "${dry_run}" == "1" ]]; then
  printf 'ssh target: %s\n' "${target}"
  printf 'remote command: %s\n' "${remote_command}"
  exit 0
fi

password_file="${JETSON_SSH_PASSWORD_FILE:-}"
if [[ -n "${password_file}" || -n "${JETSON_SSH_PASSWORD:-}" ]]; then
  if ! command -v sshpass >/dev/null 2>&1; then
    echo "sshpass is required when JETSON_SSH_PASSWORD or JETSON_SSH_PASSWORD_FILE is set." >&2
    exit 2
  fi
  if [[ -n "${password_file}" ]]; then
    exec sshpass -f "${password_file}" "${ssh_cmd[@]}"
  fi
  temp_password_file="$(mktemp)"
  chmod 600 "${temp_password_file}"
  trap 'rm -f "${temp_password_file}"' EXIT
  printf '%s\n' "${JETSON_SSH_PASSWORD}" > "${temp_password_file}"
  sshpass -f "${temp_password_file}" "${ssh_cmd[@]}"
  exit $?
fi

exec "${ssh_cmd[@]}"
