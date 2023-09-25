#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

if [ -c /dev/hm_host_pcie ]; then
  export HDPL_PLATFORM=ASIC
fi

if [ ! -d build ]; then
  mkdir build
fi

cd build || exit 1
cmake ..
make

PATH="$(pwd):${PATH}"
export PATH
cd -
hdpl_pointpillars_run
