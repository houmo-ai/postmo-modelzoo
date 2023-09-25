#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

export HDPL_PLATFORM=ISIM
python3 compile.py --pfe-model-path "${MODEL_PATH}/pointpillars_pfe.onnx" --pfe-output=pointpillars_pfe
python3 compile.py --rpn-model-path "${MODEL_PATH}/pointpillars_rpn.onnx" --rpn-output=pointpillars_rpn
