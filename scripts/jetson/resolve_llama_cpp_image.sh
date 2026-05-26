#!/usr/bin/env bash

resolve_llama_cpp_image() {
  if [[ -n "${LLAMA_CPP_DOCKER_IMAGE:-}" ]]; then
    printf '%s\n' "${LLAMA_CPP_DOCKER_IMAGE}"
    return 0
  fi

  if command -v autotag >/dev/null 2>&1; then
    local resolved_image
    if resolved_image="$(autotag llama_cpp 2>/dev/null)" && [[ -n "${resolved_image}" ]]; then
      printf '%s\n' "${resolved_image}"
      return 0
    fi
  fi

  printf '%s\n' "${LLAMA_CPP_DOCKER_IMAGE_FALLBACK:-dustynv/llama_cpp:r36.4.0}"
}

