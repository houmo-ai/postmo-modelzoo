#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd $MODELZOO_PATH/utils/tcim_perf
if [ ! -f tcim_perf ]; then
  ./build.sh
fi

cd "${SCRIPT_DIR}"

batch=4
if [ "$1" ]; then
  batch=$1
fi

python3 get_model.py
python3 build.py --batch $batch
python3 demo.py --batch $batch
