#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

if [ $HDPL_PLATFORM == "ASIC" ]; then
  ITERATION=200
  THREAD_NUM=4
else
  ITERATION=1
  THREAD_NUM=1
fi

./demo_tcim_pointpillars ${THREAD_NUM} ${ITERATION}
