#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"

# set modelzoo env
export TCIM_INC_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/include
export HDPL_INC_PATH=$HDPL_PATH/include
export TCIM_LIB_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/lib
export HDPL_LIB_PATH=$HDPL_PATH/lib
echo "HDPL_TOOLCHAIN_ITVM_INSTALL is $HDPL_TOOLCHAIN_ITVM_INSTALL"
echo "HDPL_PATH is $HDPL_PATH"

if [[ -z "${DATASETS_PATH}" ]]; then
  export DATASETS_PATH=$SCRIPT_PATH/data/datasets
fi
if [[ -z "${MODEL_PATH}" ]]; then
  export MODEL_PATH=$SCRIPT_PATH/data/models
fi

# For bash 4.4+, must not be in posix mode, may use temporary files
perf_scripts=()
while IFS='' read -r line; do perf_scripts+=("$line"); done < <(find . -type f -name "perf.sh")
for script in "${perf_scripts[@]}"; do
  echo "Run ${script}"
  bash "${script}"
done

# For bash 4.4+, must not be in posix mode, may use temporary files
eval_scripts=()
while IFS='' read -r line; do eval_scripts+=("$line"); done < <(find . -type f -name "eval.sh")
for script in "${eval_scripts[@]}"; do
  echo "Run ${script}"
  bash "${script}"
done
