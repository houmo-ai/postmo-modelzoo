#!/bin/bash
set -e

mkdir -p logs
LOG_FILE="logs/hmassist-audit-$target-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $MODELZOO_PATH/hmassist/hmassist/utils/audit.py $@ 2>&1 | tee $LOG_FILE"
python3 $MODELZOO_PATH/hmassist/hmassist/utils/audit.py $@ 2>&1 | tee $LOG_FILE
status=${PIPESTATUS[0]}
if [ $status -ne 0 ]; then
  exit $status
fi