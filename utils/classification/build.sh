#!/usr/bin/env bash

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

cd "${WORK_PATH}" || exit 1
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install
