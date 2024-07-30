#!/bin/bash
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
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

cd $MODELZOO_PATH/utils/tcim_perf/
if [ ! -f "tcim_perf" ]; then
  ./build.sh
fi
cd -

mkdir -p logs
LOG_FILE="logs/hmassist-perf-$target-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $MODELZOO_PATH/hmassist/hmassist.py perf $param 2>&1 | tee $LOG_FILE"
python3 $MODELZOO_PATH/hmassist/hmassist.py perf $param 2>&1 | tee $LOG_FILE
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
  exit $status
fi