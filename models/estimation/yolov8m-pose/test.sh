#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd $HOUMO_MODELZOO_PATH/utils/tcim_perf
if [ ! -f tcim_perf ]; then
  ./build.sh
fi

cd "${SCRIPT_DIR}"

action="${1:-all}"

declare -A action_map=(
  ["quant"]=1
  ["build"]=2
  ["perf"]=3
  ["test"]=4
  ["demo"]=5
  ["eval"]=6
  ["all"]=9
)

action_num="${action_map[$action]:-0}"

if [[ $action_num > 0 ]]; then
  python3 get_model.py
  hmquant.sh
fi
if [[ $action_num > 1 ]]; then
  hmbuild.sh
fi
if [[ $action_num > 2 ]]; then
  hmperf.sh
fi
if [[ $action_num > 3 ]]; then
  hmtest.sh --target onnx
  hmtest.sh
fi
if [[ $action_num -eq 5 || $action_num -eq 9 ]]; then
  hmdemo.sh
fi
if [[ $action_num -eq 6 || $action_num -eq 9 ]]; then
  hmeval.sh
fi
