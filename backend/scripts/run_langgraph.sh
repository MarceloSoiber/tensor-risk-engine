#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

if [[ -f ".env" ]]; then
  while IFS='=' read -r key value || [[ -n "${key}" ]]; do
    [[ -z "${key}" || "${key}" == \#* ]] && continue
    [[ "${key}" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    value="${value%$'\r'}"
    export "${key}=${value}"
  done < ".env"
fi

POSTGRES_DB="${POSTGRES_DB:-fraud_detection}"
POSTGRES_USER="${POSTGRES_USER:-fraud_user}"
POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-fraud_password}"
POSTGRES_PORT="${POSTGRES_PORT:-5432}"

export UV_PYTHON="${UV_PYTHON:-python3.12}"
export DATABASE_URL="postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@localhost:${POSTGRES_PORT}/${POSTGRES_DB}"
export LOCAL_LLM_BASE_URL="${LANGGRAPH_LOCAL_LLM_BASE_URL:-${LOCAL_LLM_BASE_URL:-http://host.docker.internal:1234/v1}}"

if [[ "${LOCAL_LLM_BASE_URL}" == "http://localhost:1234/api/v1/chat" ]]; then
  export LOCAL_LLM_BASE_URL="http://host.docker.internal:1234/v1"
elif [[ "${LOCAL_LLM_BASE_URL}" == "http://127.0.0.1:1234/api/v1/chat" ]]; then
  export LOCAL_LLM_BASE_URL="http://host.docker.internal:1234/v1"
elif [[ "${LOCAL_LLM_BASE_URL}" == "http://localhost:1234/v1" ]]; then
  export LOCAL_LLM_BASE_URL="http://host.docker.internal:1234/v1"
elif [[ "${LOCAL_LLM_BASE_URL}" == "http://127.0.0.1:1234/v1" ]]; then
  export LOCAL_LLM_BASE_URL="http://host.docker.internal:1234/v1"
fi

exec npx @langchain/langgraph-cli@latest dev "$@"
