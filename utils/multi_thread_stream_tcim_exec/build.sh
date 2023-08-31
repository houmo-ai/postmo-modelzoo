#!/usr/bin/env bash

WORK_PATH=`pwd`

rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Debug ..
make -j
make install

