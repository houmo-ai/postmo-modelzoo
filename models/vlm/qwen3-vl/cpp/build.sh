#!/usr/bin/env bash
# Build script for Qwen3-VL C++ example on Linux
if [ ! -e 3rdparty ];then
  mkdir 3rdparty
fi
if [ ! -e 3rdparty/tokenizers-cpp ];then
  cd ..
  python3 get_model.py --type hmm
  cd cpp
fi
if [ ! -e 3rdparty/eigen3 ];then
  cd 3rdparty
  wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
  unzip -q eigen-3.4.0.zip
  mv eigen-3.4.0 eigen3
  rm -rf eigen-3.4.0.zip
  cd ..
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