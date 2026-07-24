#!/usr/bin/env bash
# Build script for yolov5s on OpenHarmony (OHOS)
#
# Prerequisites:
#   export OHOS_SDK=/path/to/ohos-sdk
#   export HOUMO_TARGET=xh2
#   export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
#   export TCIM_RUNTIME_PATH=/path/to/houmo-sdk
#
# Usage:
#   ./build_OHOS.sh

set -e

if [ -z "$OHOS_SDK" ]; then
  echo "Error: OHOS_SDK environment variable is not set"
  echo "  export OHOS_SDK=/path/to/ohos-sdk"
  exit 1
fi
if [ -z "$HOUMO_EXAMPLES_PATH" ]; then
  echo "Error: HOUMO_EXAMPLES_PATH environment variable is not set"
  echo "  export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo"
  exit 1
fi
if [ -z "$TCIM_RUNTIME_PATH" ]; then
  echo "Error: TCIM_RUNTIME_PATH environment variable is not set"
  echo "  export TCIM_RUNTIME_PATH=/path/to/houmo-sdk"
  exit 1
fi
if [ "$HOUMO_TARGET" != "xh2" ]; then
  echo "Error: HOUMO_TARGET must be 'xh2'"
  echo "  export HOUMO_TARGET=xh2"
  exit 1
fi

echo "[INFO] OHOS_SDK:           $OHOS_SDK"
echo "[INFO] HOUMO_EXAMPLES_PATH: $HOUMO_EXAMPLES_PATH"
echo "[INFO] TCIM_RUNTIME_PATH:   $TCIM_RUNTIME_PATH"
echo "[INFO] HOUMO_TARGET:        $HOUMO_TARGET"

OHOS_CLANG="${OHOS_SDK}/llvm/bin/clang"
OHOS_CLANGXX="${OHOS_SDK}/llvm/bin/clang++"
OHOS_AR="${OHOS_SDK}/llvm/bin/llvm-ar"
OHOS_SYSROOT="${OHOS_SDK}/sysroot"
OHOS_TARGET="aarch64-linux-ohos"

[ ! -x "$OHOS_CLANG" ] && { echo "Error: OHOS clang not found at $OHOS_CLANG"; exit 1; }

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

rm -rf build-ohos
mkdir -p build-ohos && cd build-ohos

cmake .. \
  -DCMAKE_C_COMPILER="$OHOS_CLANG" \
  -DCMAKE_CXX_COMPILER="$OHOS_CLANGXX" \
  -DCMAKE_C_FLAGS="--target=$OHOS_TARGET --sysroot=$OHOS_SYSROOT -fPIC" \
  -DCMAKE_CXX_FLAGS="--target=$OHOS_TARGET --sysroot=$OHOS_SYSROOT -fPIC" \
  -DCMAKE_AR="$OHOS_AR" \
  -DCMAKE_SYSTEM_NAME=OHOS \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_INSTALL_PREFIX="$WORK_PATH/bin" \
  -DCMAKE_INSTALL_RPATH='$ORIGIN' \
  -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
  -DCMAKE_BUILD_TYPE=Release

make -j$(nproc)
make install

echo "================================================"
echo "[SUCCESS] Build completed!"
echo "  Binary: $WORK_PATH/bin/example_yolov5s"
echo "================================================"

