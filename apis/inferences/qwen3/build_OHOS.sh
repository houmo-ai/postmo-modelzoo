#!/usr/bin/env bash
# Build script for qwen3 on OpenHarmony (OHOS)
#
# Prerequisites:
#   export OHOS_SDK=/path/to/ohos-sdk
#   export HOUMO_TARGET=xh2
#   export HOUMO_SDK_PATH=/path/to/houmo-sdk
#   export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
#   export TCIM_RUNTIME_PATH=/path/to/houmo-sdk
#
# Usage:
#   ./build_OHOS.sh

set -e

[ -z "$OHOS_SDK" ] && { echo "Error: OHOS_SDK not set"; exit 1; }
[ -z "$HOUMO_SDK_PATH" ] && { echo "Error: HOUMO_SDK_PATH not set"; exit 1; }
[ -z "$HOUMO_EXAMPLES_PATH" ] && { echo "Error: HOUMO_EXAMPLES_PATH not set"; exit 1; }
[ -z "$TCIM_RUNTIME_PATH" ] && { echo "Error: TCIM_RUNTIME_PATH not set"; exit 1; }
[ "$HOUMO_TARGET" != "xh2" ] && { echo "Error: HOUMO_TARGET must be xh2"; exit 1; }

OHOS_CLANG="${OHOS_SDK}/llvm/bin/clang"
OHOS_CLANGXX="${OHOS_SDK}/llvm/bin/clang++"
OHOS_AR="${OHOS_SDK}/llvm/bin/llvm-ar"
OHOS_SYSROOT="${OHOS_SDK}/sysroot"
OHOS_TARGET="aarch64-linux-ohos"

[ ! -x "$OHOS_CLANG" ] && { echo "Error: OHOS clang not found"; exit 1; }

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
echo "  Binary: $WORK_PATH/bin/example_cxx_qwen3"
echo "================================================"
