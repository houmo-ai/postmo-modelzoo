#!/bin/bash
set -e

target=$HOUMO_TARGET

cd $HOUMO_MODELZOO_PATH/utils/tcim_perf/
if [ ! -f "tcim_perf" ]; then
  ./build.sh
fi
cd -

mkdir -p logs
LOG_FILE="logs/hmassist-benchmark-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $HOUMO_MODELZOO_PATH/hmassist/hmassist.py benchmark --config benchmark.yml --target $target 2>&1 | tee $LOG_FILE"
python3 $HOUMO_MODELZOO_PATH/hmassist/hmassist.py benchmark --config benchmark.yml --target $target 2>&1 | tee $LOG_FILE
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
  exit $status
fi