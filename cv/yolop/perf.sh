#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

# shellcheck source=/dev/null
source ./env.sh


cd prepare_model
./run.sh
cd ..

cd compile_model
./run.sh --batch 1
cd ..

cd ../../utils/tcimexec/
./build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -

if [ -c /dev/hmcl_feature_in ]; then
  export HDPL_PLATFORM=ASIC
fi

tcimexec --model "${SCRIPT_DIR}/compile_model/yolop" --iterations 1000
