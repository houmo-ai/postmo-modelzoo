#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

# shellcheck source=/dev/null
# source ./env.sh

cd prepare_model
./run.sh
cd ..

export HDPL_PLATFORM=ISIM

cd compile_model
./run.sh --batch 1
cd ..

cd ../../utils/multi_stream_tcim_exec/
./build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -

if [ -c /dev/hmcl_feature_in ]; then
  export HDPL_PLATFORM=ASIC
fi

if [ -z "${IS_DEBUG}" ]; then
  ITERATION=1000
else
  ITERATION=1i
fi

multi_stream_tcimexec --model "${SCRIPT_DIR}/compile_model/yolop" --iterations ${ITERATION}
