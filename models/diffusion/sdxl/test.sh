#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

pip3 install -r requirements.txt
python3 get_model.py
python3 build.py
python3 get_model.py --type raw
python3 demo.py
