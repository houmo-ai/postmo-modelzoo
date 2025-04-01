#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd $HOUMO_MODELZOO_PATH/utils/tcim_perf
if [ ! -f tcim_perf ]; then
  ./build.sh
fi

cd "${SCRIPT_DIR}"


python3 get_model.py --type quant
python3 build.py
python3 demo.py
