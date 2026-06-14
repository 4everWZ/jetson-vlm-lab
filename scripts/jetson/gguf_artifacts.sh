#!/usr/bin/env bash

gguf_artifact_has_magic() {
  local path="$1"
  [[ -f "${path}" && "$(head -c 4 "${path}" 2>/dev/null || true)" == "GGUF" ]]
}

require_gguf_artifact() {
  local path="$1"
  local role="${2:-artifact}"
  if [[ ! -f "${path}" ]]; then
    echo "${role} GGUF not found: ${path}" >&2
    return 2
  fi
  if ! gguf_artifact_has_magic "${path}"; then
    echo "${role} GGUF failed magic check: ${path}" >&2
    echo "Expected the file to start with GGUF magic bytes. Remove the artifact and rerun the launcher to download or restore it." >&2
    return 2
  fi
}
