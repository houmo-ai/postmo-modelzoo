#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

# shellcheck source=/dev/null
# source ./env.sh

cd prepare_model
./run.sh
cd ..

cd compile_model
./run.sh --batch 4 --mode perf
cd ..

cd ../../utils/aottcimexec/
./build.sh
PATH="$(pwd):${PATH}"
export PATH
export HDPL_STREAM_TIME_OUT=600000
cd -

if [ -z "${IS_DEBUG}" ]; then
  ITERATION=1000
else
  ITERATION=1
fi

tcimexec --model "${SCRIPT_DIR}/compile_model/tcim_efficientnet" --iterations ${ITERATION}
