#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

python3 rm_onnx_postproc.py
python3 compile.py "$@"
