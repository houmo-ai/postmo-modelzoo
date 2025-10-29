#!/usr/bin/env bash
# c++ example
if [ $(uname -s) = "Linux" ] && [ $(uname -m) = "x86_64" ]; then
  if [ "$HOUMO_TARGET" = "xh2" ]; then
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    # get test model
    python3 get_model.py
    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release ..
    make
    make install

    cd $WORK_PATH
    ./example_cxx_qwen3
  elif [ "$HOUMO_TARGET" = "xh1" ]; then
    echo "Xh1 Environment"
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    # get test model
    python3 get_model.py
    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_CXX_FLAGS="-DBACKEND_XH1" -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release .. 
    make
    make install

    cd $WORK_PATH
    ./example_cxx_qwen3
  else
    echo "UnSupport Backend!"
  fi
elif [ $(uname -s) = "Linux" ] && [ $(uname -m) = "aarch64" ]; then
  if [ "$HOUMO_TARGET" = "xh1" ]; then
    echo "Xh1 Environment"
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    # get test model
    python3 get_model.py
    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_CXX_FLAGS="-DBACKEND_XH1" -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DCMAKE_BUILD_TYPE=Release .. 
    make
    make install

    cd $WORK_PATH
    ./example_cxx_qwen3
  else
    echo "UnSupport Backend!"
  fi
else
  echo "UnSupport PlatForm!"
fi