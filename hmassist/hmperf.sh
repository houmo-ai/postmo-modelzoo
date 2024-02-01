#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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

cd $MODELZOO_PATH/utils/aottcimexec/
if [ ! -f "tcimexec" ]; then
  ./build.sh
fi
cd -

mkdir -p logs
LOG_FILE="logs/hmassist-perf-$target-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $MODELZOO_PATH/hmassist/hmassist.py perf $param 2>&1 | tee $LOG_FILE"
python3 $MODELZOO_PATH/hmassist/hmassist.py perf $param 2>&1 | tee $LOG_FILE