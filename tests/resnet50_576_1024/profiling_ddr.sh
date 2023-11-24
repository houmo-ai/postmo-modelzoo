#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

cd prepare_model
./run.sh
cd ..

cd compile_model
#./run.sh --batch 1
cd ..

HDPL_MODULE_PATH=$(python3 -c 'import hdpl;print(hdpl.__path__[0])')
cd "${HDPL_MODULE_PATH}/tools/aot_profiler"
if [ ! -f gen_profile_data ]; then
  cmake .
  make
fi
export HDPL_STREAM_TIME_OUT=150000
if [ -d output ]; then
  rm -rf output
fi
./profile_bandwidth.sh "${SCRIPT_DIR}/compile_model/tcim_resnet50"
cd -
