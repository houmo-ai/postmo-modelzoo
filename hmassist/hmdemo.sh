#!/bin/bash
set -e

target=$HMASSIST_TARGET

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
LOG_FILE="logs/hmassist-demo-$target-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $MODELZOO_PATH/hmassist/hmassist.py demo $param 2>&1 | tee $LOG_FILE"
python3 $MODELZOO_PATH/hmassist/hmassist.py demo $param 2>&1 | tee $LOG_FILE