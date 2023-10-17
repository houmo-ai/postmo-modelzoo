#!/bin/bash

log_highlight() {
  echo -e "\e[30;31m"$1"$(tput sgr0)"
}

mkdir -p logs
LOG_FILE="logs/hmassist-build-H30-$(date "+%Y-%m-%d-%H-%M-%S").log"

echo "python3 $MODELZOO_PATH/hmassist/hmassist.py build --target H30 -c config.yml 2>&1 | tee $LOG_FILE"
python3 $MODELZOO_PATH/hmassist/hmassist.py build --target H30 -c config.yml 2>&1 | tee $LOG_FILE