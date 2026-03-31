#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

# Required env vars for integration tests:
#   HICT_JVM_API_BASE_URL
# Optional env vars:
#   HICT_DATASET_FILE
#   HICT_FASTA_FILE
#   HICT_AGP_FILE
#   HICT_JVM_API_ALLOW_MUTATION=true
pytest -vv "$SCRIPT_DIR/tests_jvm_api/test_optional_integration.py" -m integration
