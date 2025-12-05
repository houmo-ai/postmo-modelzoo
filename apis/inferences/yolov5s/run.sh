#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [ -z "$houmo_target" ] || [ "$houmo_target" != "xh2" ]; then
    echo "Only supports HOUMO_TARGET as xh2."
    exit 1
fi

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
    export LD_LIBRARY_PATH=$WORK_PATH/../../models/3rdparty/onnxruntime/lib:$LD_LIBRARY_PATH
    ./example_yolov5s --enable_ort
else
    ./example_yolov5s
fi
