#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh1" ]; then
    echo "Only supports HOUMO_TARGET as xh1."
    exit 0
fi

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

# get test model
python3 get_model.py

# c++ example
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
make
make install

cd $WORK_PATH
./example_yolov5s_dynamic

