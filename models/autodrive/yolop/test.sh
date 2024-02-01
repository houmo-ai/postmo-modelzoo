#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

if [ ! -f "resnet50.onnx" ]; then
  python3 get_model.py --type raw
fi

# python test
python3 ptq.py
python3 build.py