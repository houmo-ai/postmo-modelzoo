#!/usr/bin/env bash
# Build script for tcim_perf on OpenHarmony (OHOS)
#
# Prerequisites:
#   export OHOS_SDK=/path/to/ohos-sdk
#   export HOUMO_TARGET=xh2
#   export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
#   export TCIM_RUNTIME_PATH=/path/to/houmo-sdk
#   export HOUMO_SDK_PATH=/path/to/houmo-sdk
#
# Usage:
#   ./build_OHOS.sh

set -e

# ------------------------------------------------------------------
# Check environment
# ------------------------------------------------------------------
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

if [ -z "$HOUMO_SDK_PATH" ]; then
  echo "Error: HOUMO_SDK_PATH environment variable is not set"
  echo "  export HOUMO_SDK_PATH=/path/to/houmo-sdk"
  exit 1
fi

echo "[INFO] OHOS_SDK:           $OHOS_SDK"
echo "[INFO] HOUMO_SDK_PATH:      $HOUMO_SDK_PATH"
echo "[INFO] HOUMO_EXAMPLES_PATH: $HOUMO_EXAMPLES_PATH"

OHOS_CLANG="${OHOS_SDK}/llvm/bin/clang"
OHOS_CLANGXX="${OHOS_SDK}/llvm/bin/clang++"
OHOS_AR="${OHOS_SDK}/llvm/bin/llvm-ar"
OHOS_SYSROOT="${OHOS_SDK}/sysroot"
OHOS_TARGET="aarch64-linux-ohos"

if [ ! -x "$OHOS_CLANG" ]; then
  echo "Error: OHOS clang not found at $OHOS_CLANG"
  exit 1
fi

# ------------------------------------------------------------------
# CMake cross-compilation
# ------------------------------------------------------------------
WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

echo "[INFO] Configuring CMake..."

rm -rf build-ohos
mkdir -p build-ohos
cd build-ohos || exit 1

cmake .. \
  -DCMAKE_C_COMPILER="$OHOS_CLANG" \
  -DCMAKE_CXX_COMPILER="$OHOS_CLANGXX" \
  -DCMAKE_C_FLAGS="--target=$OHOS_TARGET --sysroot=$OHOS_SYSROOT -fPIC" \
  -DCMAKE_CXX_FLAGS="--target=$OHOS_TARGET --sysroot=$OHOS_SYSROOT -fPIC" \
  -DCMAKE_AR="$OHOS_AR" \
  -DCMAKE_SYSTEM_NAME=OHOS \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_INSTALL_PREFIX="$WORK_PATH/../bin" \
  -DCMAKE_BUILD_TYPE=Release

echo "[INFO] Building..."
make -j$(nproc)

echo "[INFO] Installing..."
make install

echo "================================================"
echo "[SUCCESS] Build completed!"
echo "  Binary: $WORK_PATH/../bin/tcim_perf"
echo "================================================"
