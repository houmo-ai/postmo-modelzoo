#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

cd prepare_model
./run.sh
cd ..

cd compile_model
./run.sh --batch 24
cd ..

cd ../../utils/aottcimexec/
./build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -

if [ -c /dev/hm_host_pcie ]; then
  export HDPL_PLATFORM=ASIC
fi

if [ -z "${IS_DEBUG}" ]; then
  ITERATION=1000
else
  ITERATION=1
fi

export HDPL_STREAM_TIME_OUT=150000
tcimexec --model "${SCRIPT_DIR}/compile_model/tcim_resnet50" --iterations ${ITERATION}
