#!/bin/bash
set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
export PYTHONPATH="$SCRIPT_DIR:${PYTHONPATH:-}"

pytest -vv "$SCRIPT_DIR/tests_jvm_api" -m "not integration"
