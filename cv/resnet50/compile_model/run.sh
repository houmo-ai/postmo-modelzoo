#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

python3 calibrate.py
export SMPLAN_RESIZER_PARALLEL=1
python3 compile.py
