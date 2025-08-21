#!/usr/bin/env bash
set -e

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

RESIZE_TYPE=0
if [ "$1" = "1" ]; then
  RESIZE_TYPE=1
fi

if [[ -z $HOUMO_EXAMPLES_PATH ]]; then
  export HOUMO_EXAMPLES_PATH=$WORK_PATH/../../..
fi

# get example models and data
python3 get_model.py

# c++ example
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install

cd $WORK_PATH
./multibatch_example $RESIZE_TYPE
