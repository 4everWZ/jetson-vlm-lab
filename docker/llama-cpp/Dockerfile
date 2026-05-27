# syntax=docker/dockerfile:1

ARG BASE_RUNTIME=dustynv/cuda-python:r36.4.0-cu128-24.04

FROM ${BASE_RUNTIME} AS runtime

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG LLAMA_CPP_REF=unknown

LABEL org.opencontainers.image.title="jetson-llama-cpp"
LABEL org.opencontainers.image.description="Jetson Orin ARM64 llama.cpp CUDA runtime with official llama-server/mmproj support"
LABEL org.opencontainers.image.source="https://github.com/4everWZ/jetson-vlm-lab"
LABEL org.opencontainers.image.base.name="dustynv/cuda-python:r36.4.0-cu128-24.04"
LABEL org.opencontainers.image.vendor="4everWZ"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.version="${LLAMA_CPP_REF}"

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 \
    ca-certificates \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY artifacts/llama.cpp-install /opt/llama.cpp

ENV PATH="/opt/llama.cpp/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/targets/aarch64-linux/lib:${LD_LIBRARY_PATH}"

WORKDIR /workspace

CMD ["llama-server", "--help"]
