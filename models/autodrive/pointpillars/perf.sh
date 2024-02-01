#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

python3 get_model.py
python3 build_rpn.py --stage build
python3 build_pfe_1.py --stage build

cd cpp
./build.sh
./run.sh
