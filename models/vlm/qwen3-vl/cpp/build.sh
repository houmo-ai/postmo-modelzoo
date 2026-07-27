#!/usr/bin/env bash
# Build script for Qwen3-VL C++ example on Linux
if [ ! -e 3rdparty ];then
  mkdir 3rdparty
fi
if [ ! -e 3rdparty/tokenizers-cpp ];then
  python3 get_3rdparty.py
fi
if [ $(uname -s) = "Linux" ] &&  ([ $(uname -m) = "x86_64" ] || [ $(uname -m) = "aarch64" ]); then
  if [ "$HOUMO_TARGET" = "xh2" ]; then
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH/../bin -DCMAKE_BUILD_TYPE=Release ..
    make
    make install
  else
    echo "UnSupport Backend!"
  fi
else
  echo "UnSupport PlatForm!"
fi