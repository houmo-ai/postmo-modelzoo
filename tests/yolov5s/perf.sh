#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

# shellcheck source=/dev/null

python3 store_yolov5s.py

cd ../../utils/aottcimexec/
if [ ! -f tcimexec ]; then
  ./build.sh
fi
PATH="$(pwd):${PATH}"
export PATH
cd -

if [ -c /dev/hmcl_feature_in ]; then
  export HDPL_PLATFORM=ASIC
fi

if [ -z "${IS_DEBUG}" ]; then
  ITERATION=1000
else
  ITERATION=1
fi

tcimexec --model "${SCRIPT_DIR}/tcim_yolov5s" --iterations ${ITERATION}
