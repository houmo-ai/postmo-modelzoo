#!/bin/bash
set -e

target=$HOUMO_TARGET

found_target=false
argc=$#
for ((i=1; i<=$argc; i++)); do
  arg="${!i}"
  if [ "$arg" == "--target" ]; then
    param=$@
    target=$((i+1))
    found_target=true
    break
  fi
done

if [ "$found_target" == false ]; then
  param="--target $target $@"
fi

mkdir -p logs
LOG_FILE="logs/hmassist-quant-$target-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $MODELZOO_PATH/hmassist/hmassist.py quant $param 2>&1 | tee $LOG_FILE"
python3 $MODELZOO_PATH/hmassist/hmassist.py quant $param 2>&1 | tee $LOG_FILE
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
  exit $status
fi