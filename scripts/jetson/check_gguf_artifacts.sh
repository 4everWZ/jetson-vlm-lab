#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}" python3 -m edge_vlm.gguf_artifacts "$@"
