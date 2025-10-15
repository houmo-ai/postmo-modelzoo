#!/usr/bin/env bash
set -e

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

# build c++ example
rm -rf build
mkdir -p build
cd build || exit 1

cmake -DCMAKE_BUILD_TYPE=Release ..
make -j

cd $WORK_PATH
rm -rf bin
mkdir -p bin
cp ./build/dev_monitor ./bin/