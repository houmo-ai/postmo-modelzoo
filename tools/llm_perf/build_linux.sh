#!/usr/bin/env bash
# c++ example
if [ ! -e 3rdparty/eigen3 ];then
  wget ${HOUMO_MODELZOO_URL}/models/qwen3/3rdparty.zip
  unzip 3rdparty.zip
  rm -rf 3rdparty.zip
fi
if [ $(uname -s) = "Linux" ] && [ $(uname -m) = "x86_64" ]; then
  if [ "$HOUMO_TARGET" = "xh2" ]; then
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH/../bin -DCMAKE_BUILD_TYPE=Release ..
    make
    make install
  elif [ "$HOUMO_TARGET" = "xh1" ]; then
    echo "Xh1 Environment"
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    # get test model
    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_CXX_FLAGS="-DBACKEND_XH1" -DCMAKE_INSTALL_PREFIX=$WORK_PATH/../bin -DCMAKE_BUILD_TYPE=Release .. 
    make
    make install
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
    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_CXX_FLAGS="-DBACKEND_XH1" -DCMAKE_INSTALL_PREFIX=$WORK_PATH/../bin -DCMAKE_BUILD_TYPE=Release .. 
    make
    make install
  else
    echo "UnSupport Backend!"
  fi
else
  echo "UnSupport PlatForm!"
fi