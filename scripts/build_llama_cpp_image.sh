#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

LLAMA_CPP_REF="${LLAMA_CPP_REF:-$(git ls-remote https://github.com/ggml-org/llama.cpp.git HEAD | awk '{print $1}')}"
IMAGE_TAG="${IMAGE_TAG:-ghcr.io/4everwz/jetson-llama-cpp:r36.4-cu128-u24.04-sm87-${LLAMA_CPP_REF:0:7}}"
BUILD_DATE="${BUILD_DATE:-$(date -u +%Y-%m-%dT%H:%M:%SZ)}"
VCS_REF="${VCS_REF:-$(git rev-parse --short=12 HEAD)}"
SKIP_ARTIFACT_BUILD="${SKIP_ARTIFACT_BUILD:-0}"
DOCKER_BIN="${DOCKER_BIN:-}"

if [[ -n "${DOCKER_BIN}" ]]; then
  read -r -a DOCKER_CMD <<< "${DOCKER_BIN}"
elif docker ps >/dev/null 2>&1; then
  DOCKER_CMD=(docker)
else
  DOCKER_CMD=(sudo docker)
fi

echo "LLAMA_CPP_REF=${LLAMA_CPP_REF}"
echo "IMAGE_TAG=${IMAGE_TAG}"
echo "BUILD_DATE=${BUILD_DATE}"
echo "VCS_REF=${VCS_REF}"
echo "DOCKER_CMD=${DOCKER_CMD[*]}"

if [[ "${SKIP_ARTIFACT_BUILD}" == "0" ]]; then
  LLAMA_CPP_REF="${LLAMA_CPP_REF}" scripts/build_llama_cpp_artifacts.sh
elif [[ "${SKIP_ARTIFACT_BUILD}" != "1" ]]; then
  echo "SKIP_ARTIFACT_BUILD must be 0 or 1." >&2
  exit 2
fi

if [[ ! -x artifacts/llama.cpp-install/bin/llama-server ]]; then
  echo "Missing artifacts/llama.cpp-install/bin/llama-server; run scripts/build_llama_cpp_artifacts.sh first." >&2
  exit 2
fi

"${DOCKER_CMD[@]}" build \
  -f docker/llama-cpp/Dockerfile \
  --build-arg "BUILD_DATE=${BUILD_DATE}" \
  --build-arg "VCS_REF=${VCS_REF}" \
  --build-arg "LLAMA_CPP_REF=${LLAMA_CPP_REF}" \
  -t "${IMAGE_TAG}" \
  .

echo "Built ${IMAGE_TAG}"
