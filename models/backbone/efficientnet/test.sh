#!/usr/bin/env bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${SCRIPT_DIR}"

python3 get_model.py
python3 ptq.py
python3 build.py
hmdemo.sh
