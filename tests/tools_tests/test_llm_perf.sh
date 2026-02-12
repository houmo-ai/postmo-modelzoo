#!/usr/bin/env bash
# run llm_perf test
if [ $(uname -s) = "Linux" ] && [ $(uname -m) = "x86_64" ]; then
  if [ "$HOUMO_TARGET" = "xh2" ]; then
    set -e

    cd $HOUMO_EXAMPLES_PATH/tests/tools_tests/
    export HOUMO_MODELZOO_URL=http://artifactory.houmo.ai/artifactory/Dadao
    python3 test_perf_models.py
  else
    echo "UnSupport Backend!"
  fi
else
  echo "UnSupport PlatForm!"
fi