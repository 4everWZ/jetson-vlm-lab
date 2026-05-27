#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

BUILDER_IMAGE="${BUILDER_IMAGE:-dustynv/cuda:12.8-samples-r36.4.0-cu128-24.04}"
LLAMA_CPP_REF="${LLAMA_CPP_REF:-$(git ls-remote https://github.com/ggml-org/llama.cpp.git HEAD | awk '{print $1}')}"
BUILD_JOBS="${BUILD_JOBS:-2}"

OUT_DIR="$PWD/artifacts/llama.cpp-install"

echo "BUILDER_IMAGE=$BUILDER_IMAGE"
echo "LLAMA_CPP_REF=$LLAMA_CPP_REF"
echo "BUILD_JOBS=$BUILD_JOBS"
echo "OUT_DIR=$OUT_DIR"

rm -rf "$OUT_DIR"
mkdir -p "$OUT_DIR"

sudo docker run --rm \
  --runtime nvidia \
  --network host \
  --ipc host \
  -e LLAMA_CPP_REF="$LLAMA_CPP_REF" \
  -e BUILD_JOBS="$BUILD_JOBS" \
  -v "$PWD/artifacts:/artifacts" \
  "$BUILDER_IMAGE" \
  bash -lc '
set -eux

apt-get update
apt-get install -y --no-install-recommends \
  git \
  cmake \
  build-essential \
  curl \
  libcurl4-openssl-dev \
  ca-certificates \
  pkg-config \
  libgomp1

rm -rf /tmp/llama.cpp
git clone https://github.com/ggml-org/llama.cpp.git /tmp/llama.cpp
cd /tmp/llama.cpp
git checkout "$LLAMA_CPP_REF"

cmake -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_CUDA=ON \
  -DGGML_CUDA_F16=ON \
  -DGGML_CUDA_FA_ALL_QUANTS=OFF \
  -DLLAMA_CURL=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_EXAMPLES=OFF \
  -DLLAMA_BUILD_TOOLS=ON \
  -DLLAMA_BUILD_SERVER=ON \
  -DBUILD_SHARED_LIBS=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=87

cmake --build build --config Release --target llama-server llama-mtmd-cli -- -j"$BUILD_JOBS"

mkdir -p /artifacts/llama.cpp-install/bin
cp -av build/bin/llama-server /artifacts/llama.cpp-install/bin/
cp -av build/bin/llama-mtmd-cli /artifacts/llama.cpp-install/bin/

echo "$LLAMA_CPP_REF" > /artifacts/llama.cpp-install/LLAMA_CPP_REF

ldd /artifacts/llama.cpp-install/bin/llama-server || true
/artifacts/llama.cpp-install/bin/llama-server --help | grep -n -C 5 -E -- "--mmproj|mmproj|mtmd|image" || true
'

echo "Artifacts built:"
find "$OUT_DIR" -maxdepth 3 -type f -print
