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
./run.sh --batch 1
cd ..

cd ../../utils/threadtcimexec/
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
  ITERATION=1
fi

export HDPL_STREAM_TIME_OUT=150000
e2etcimexec --model "${SCRIPT_DIR}/compile_model/tcim_yolop" --iterations "${ITERATION}" "$@"
