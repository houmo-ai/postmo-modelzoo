#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd $MODELZOO_PATH/utils/tcim_perf
if [ ! -f tcim_perf ]; then
  ./build.sh
fi

cd "${SCRIPT_DIR}"

python3 get_model.py --type quant
hmbuild.sh
hmperf.sh
