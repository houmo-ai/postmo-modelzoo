#!/usr/bin/env bash
set -e

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

if [[ -z $HOUMO_EXAMPLES_PATH ]]; then
  export HOUMO_EXAMPLES_PATH=$WORK_PATH/../..
fi

# get test model
python3 get_model.py

# c++ example
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install

export LD_LIBRARY_PATH=$HOUMO_PATH/lib:$HOUMO_SDK_PATH/hal/lib:$HOUMO_EXAMPLES_PATH/3rdparty/install/ffmpeg/lib:$LD_LIBRARY_PATH
cd $WORK_PATH
./example_video_detect
