#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

export HDPL_PLATFORM=ISIM
export MEM_PLAN_DUMP=1
export SKIP_CORE=1
python3 store_bev2d.py --batch 1
unset MEM_PLAN_DUMP
unset SKIP_CORE

cd ../../utils/tcimexec/
./build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -

if [ -c /dev/hm_host_pcie ]; then
  export HDPL_PLATFORM=ASIC
fi

if [ -z "${IS_DEBUG}" ]; then
  ITERATION=1000
else
  ITERATION=1
fi

if [ -f /opt/sys ]; then
  pushd /opt/sys/
    ./reset_aicore.sh
  popd
fi

tcimexec --model "${SCRIPT_DIR}/bev2d" --iterations ${ITERATION} --host_loop
