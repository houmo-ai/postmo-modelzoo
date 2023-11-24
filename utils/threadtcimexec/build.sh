#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

rm -rf build
mkdir -p build
cd build || exit 1
cmake "-DCMAKE_INSTALL_PREFIX=${SCRIPT_DIR}" -DCMAKE_BUILD_TYPE=Debug ..
make -j
make install
