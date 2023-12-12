#!/bin/bash

target=H30
if [ "$1" ]; then
  target=$1
fi

mkdir -p logs
LOG_FILE="logs/hmassist-quant-$target-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $MODELZOO_PATH/hmassist/hmassist.py quant --target $target 2>&1 | tee $LOG_FILE"
python3 $MODELZOO_PATH/hmassist/hmassist.py quant --target $target 2>&1 | tee $LOG_FILE