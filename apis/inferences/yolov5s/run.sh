#!/usr/bin/env bash
set -e

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

ORT_SWITCH=OFF
if [ "$1" = "enable_ort" ]; then
  ORT_SWITCH=ON
fi

if [ "$ORT_SWITCH" = "ON" ]; then
    # get test model
    python3 get_model.py --enable_ort

    # python example
    python3 yolov5s.py --enable_ort
else
    # get test model
    python3 get_model.py

    # python example
    python3 yolov5s.py
fi

# c++ example
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DENABLE_ORT=$ORT_SWITCH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install

cd $WORK_PATH
if [ "$ORT_SWITCH" = "ON" ]; then
    ./example_yolov5s --enable_ort
else
    ./example_yolov5s
fi
