#!/usr/bin/env bash

WORK_PATH=`pwd`

mkdir -p build
cd build
cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install
