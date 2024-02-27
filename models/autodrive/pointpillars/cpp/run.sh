#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

if [ $HDPL_PLATFORM == "ASIC" ]; then
  ITERATION=200
else
  ITERATION=1
fi

./demo_tcim_pointpillars ${ITERATION}
