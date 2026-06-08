#!/usr/bin/env bash

download_hf_file() {
  local repo_id="$1"
  local filename="$2"
  local destination="$3"
  if [[ -f "${destination}" ]]; then
    return 0
  fi
  if ! command -v curl >/dev/null 2>&1; then
    echo "curl is required to download ${repo_id}/${filename}; pre-place the file or install curl." >&2
    exit 2
  fi
  local url="https://huggingface.co/${repo_id}/resolve/main/${filename}"
  local partial="${destination}.partial"
  local curl_log="${partial}.curl.log"
  echo "Downloading ${url} -> ${destination}" >&2
  local curl_status=0
  curl --fail --location --retry 3 --continue-at - --output "${partial}" "${url}" 2> >(tee "${curl_log}" >&2) || curl_status=$?
  if [[ "${curl_status}" -ne 0 ]]; then
    if [[ "${curl_status}" -eq 22 && -f "${partial}" ]] \
      && grep -q "416" "${curl_log}" \
      && [[ "$(head -c 4 "${partial}" 2>/dev/null || true)" == "GGUF" ]]; then
      echo "HTTP 416 while resuming ${destination}; accepting existing GGUF partial as complete." >&2
    else
      rm -f "${curl_log}"
      return "${curl_status}"
    fi
  fi
  rm -f "${curl_log}"
  mv "${partial}" "${destination}"
}
