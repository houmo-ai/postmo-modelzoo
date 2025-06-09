#!/usr/bin/env bash
set -e

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

if [[ -z $HOUMO_EXAMPLES_PATH ]]; then
  export HOUMO_EXAMPLES_PATH=$WORK_PATH/../..
fi

RESIZER_SWITCH=ON
if [ "$1" = "disable_resizer" ]; then
  RESIZER_SWITCH=OFF
fi

# get test model
python3 get_model.py

# c++ example
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DENABLE_RESIZER=$RESIZER_SWITCH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install

cd $WORK_PATH
./example_video_detect
