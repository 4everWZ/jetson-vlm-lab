#!/usr/bin/env bash
set -Eeuo pipefail

host="${VLM_SERVER_HOST:-127.0.0.1}"
port="${VLM_SERVER_PORT:-8080}"
base_url="${VLM_SERVER_BASE_URL:-http://${host}:${port}}"

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to check llama-server health." >&2
  exit 2
fi

echo "Checking ${base_url}/v1/health"
curl --fail --silent --show-error "${base_url}/v1/health" >/dev/null

echo "Checking ${base_url}/v1/models"
curl --fail --silent --show-error "${base_url}/v1/models"
echo
