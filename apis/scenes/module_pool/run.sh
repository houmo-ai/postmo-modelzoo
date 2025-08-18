#!/usr/bin/env bash
set -e

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

# get test model
python3 get_model.py

# build c++ example
rm -rf build
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install

cd $WORK_PATH
./example_module_pool
