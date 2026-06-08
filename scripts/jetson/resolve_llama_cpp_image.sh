#!/usr/bin/env bash

resolve_llama_cpp_image() {
  if [[ -n "${LLAMA_CPP_DOCKER_IMAGE:-}" ]]; then
    printf '%s\n' "${LLAMA_CPP_DOCKER_IMAGE}"
    return 0
  fi

  # The dusty-nv llama_cpp image has not provided the multimodal llama-server
  # path this repo needs. Keep autotag opt-in so default VLM launchers use the
  # verified self-built official llama.cpp image instead.
  if [[ "${LLAMA_CPP_USE_AUTOTAG:-0}" == "1" ]] && command -v autotag >/dev/null 2>&1; then
    local resolved_image
    if resolved_image="$(autotag llama_cpp 2>/dev/null)" && [[ -n "${resolved_image}" ]]; then
      printf '%s\n' "${resolved_image}"
      return 0
    fi
  fi

  printf '%s\n' "${LLAMA_CPP_DOCKER_IMAGE_FALLBACK:-ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87}"
}
