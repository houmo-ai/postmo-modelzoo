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

cd ../../utils/multi_stream_tcim_exec/
./build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -

if [ -z "${IS_DEBUG}" ]; then
  ITERATION=200
else
  ITERATION=1
fi

multi_stream_tcimexec --model "${SCRIPT_DIR}/compile_model/efficientnet" --iterations ${ITERATION}
