#!/usr/bin/env bash
# c++ example
export HOUMO_ENGINE_DIR="$HOUMO_EXAMPLES_PATH/tools/common/houmo-llm-engine"
export CURRENT_DIR=$(pwd)
cd $HOUMO_ENGINE_DIR
if [ -e build ]; then
  rm -rf build
fi
./build_linux.sh
cd $CURRENT_DIR
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
echo "Build successfully!"