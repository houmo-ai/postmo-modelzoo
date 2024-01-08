#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

python3 get_model.py --type raw

# python test
python3 ptq.py
python3 build.py

# hmassist test
hmquant.sh
hmbuild.sh
hmtest.sh
hmdemo.sh
hmperf.sh
hmaccuracy.sh