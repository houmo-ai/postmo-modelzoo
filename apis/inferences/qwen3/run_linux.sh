#!/usr/bin/env bash
# c++ example
if [ $(uname -s) = "Linux" ] && [ $(uname -m) = "x86_64" ]; then
  if [ "$HOUMO_TARGET" = "xh2" ]; then
    set -e
    if [ ! -e 3rdparty ];then
      mkdir 3rdparty
    fi
    if [ ! -e 3rdparty/eigen3 ];then
      cd 3rdparty
      wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip
      unzip eigen-3.4.0.zip
      mv eigen-3.4.0 eigen3
      rm -rf eigen-3.4.0.zip
      cd ..
    fi
    if [ ! -e 3rdparty/tokenizers-cpp ];then
      cd 3rdparty
      wget ${HOUMO_MODELZOO_URL}/3rdparty/qwen3-tokenizers-cpp.zip
      unzip qwen3-tokenizers-cpp.zip
      rm -rf qwen3-tokenizers-cpp.zip
      cd ..
    fi

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
  else
    echo "UnSupport Backend!"
  fi
elif [ $(uname -s) = "Linux" ] && [ $(uname -m) = "aarch64" ]; then
  echo "UnSupport Backend!"
else
  echo "UnSupport PlatForm!"
fi