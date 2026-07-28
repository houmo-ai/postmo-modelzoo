#!/usr/bin/env bash
# Build script for CosyVoice3 C++ demo on OpenHarmony (OHOS)
#
# Prerequisites:
#   export OHOS_SDK=/path/to/ohos-sdk
#   export HOUMO_TARGET=xh2
#   export HOUMO_EXAMPLES_PATH=/path/to/imodelzoo
#   export TCIM_RUNTIME_PATH=/path/to/houmo-sdk
#
# Usage:
#   ./build_OHOS.sh                    # use prebuilt audio libs
#   ./build_OHOS.sh --audio-source     # build audio libs from source
#   ./build_OHOS.sh --audio-prebuilt   # use prebuilt audio libs

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

if [ ! -x "$OHOS_CLANG" ]; then
  echo "Error: OHOS clang not found at $OHOS_CLANG"
  exit 1
fi

# ------------------------------------------------------------------
# Parse arguments
# ------------------------------------------------------------------
BUILD_AUDIO_FROM_SOURCE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        --audio-source)
            BUILD_AUDIO_FROM_SOURCE="-DBUILD_AUDIO_FROM_SOURCE=ON"
            shift
            ;;
        --audio-prebuilt)
            BUILD_AUDIO_FROM_SOURCE=""
            shift
            ;;
        *)
            shift
            ;;
    esac
done

# ------------------------------------------------------------------
# Setup paths
# ------------------------------------------------------------------
WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

if [ ! -e 3rdparty ]; then
  mkdir 3rdparty
fi

# ------------------------------------------------------------------
# Tokenizers-cpp (check for OHOS-compatible prebuilt libs)
# ------------------------------------------------------------------
TOKENIZERS_OHOS_DIR="3rdparty/tokenizers-cpp/OHOS_aarch64_xh2"
if [ ! -e "$TOKENIZERS_OHOS_DIR/lib/libtokenizers_cpp.a" ] || \
   [ ! -e "$TOKENIZERS_OHOS_DIR/lib/libtokenizers_c.a" ]; then
  echo "[INFO] Setting up tokenizers-cpp for OHOS..."
  mkdir -p "$TOKENIZERS_OHOS_DIR/include"
  mkdir -p "$TOKENIZERS_OHOS_DIR/lib"

  # Copy headers from source
  if [ -e 3rdparty/tokenizers-cpp/include/tokenizers_cpp.h ]; then
    cp 3rdparty/tokenizers-cpp/include/tokenizers_cpp.h "$TOKENIZERS_OHOS_DIR/include/"
  fi

  # Try to find prebuilt .a from whisper build
  WHISPER_TOKENS="$(dirname "$WORK_PATH")/../../asr/whisper/cpp/3rdparty/tokenizers-cpp/build-ohos"
  if [ -e "$WHISPER_TOKENS/libtokenizers_cpp.a" ] && \
     [ -e "$WHISPER_TOKENS/libtokenizers_c.a" ]; then
    cp "$WHISPER_TOKENS/libtokenizers_cpp.a" "$WHISPER_TOKENS/libtokenizers_c.a" \
       "$TOKENIZERS_OHOS_DIR/lib/"
    echo "[INFO] Copied tokenizers-cpp .a from whisper build"
  else
    echo "[WARN] OHOS tokenizers-cpp libs not found"
    echo "       Please cross-compile tokenizers-cpp for OHOS first, then copy:"
    echo "         cp build-ohos/libtokenizers_cpp.a build-ohos/libtokenizers_c.a \\"
    echo "            $TOKENIZERS_OHOS_DIR/lib/"
  fi
fi

# ------------------------------------------------------------------
# CMake cross-compilation
# ------------------------------------------------------------------
BUILD_DIR="build-ohos"
echo "[INFO] Configuring CMake..."

rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR" || exit 1

cmake .. \
  -DCMAKE_C_COMPILER="$OHOS_CLANG" \
  -DCMAKE_CXX_COMPILER="$OHOS_CLANGXX" \
  -DCMAKE_C_FLAGS="--target=$OHOS_TARGET --sysroot=$OHOS_SYSROOT -fPIC" \
  -DCMAKE_CXX_FLAGS="--target=$OHOS_TARGET --sysroot=$OHOS_SYSROOT -fPIC" \
  -DCMAKE_AR="$OHOS_AR" \
  -DCMAKE_SYSTEM_NAME=OHOS \
  -DCMAKE_SYSTEM_PROCESSOR=aarch64 \
  -DCMAKE_INSTALL_PREFIX="$WORK_PATH/../bin" \
  -DCMAKE_BUILD_TYPE=Release \
  ${BUILD_AUDIO_FROM_SOURCE}

echo "[INFO] Building..."
make -j$(nproc)

echo "[INFO] Installing..."
make install

echo "================================================"
echo "[SUCCESS] Build completed!"
echo "  Binary:    $WORK_PATH/../bin/cosyvoice3-demo"
echo "  Libraries: $WORK_PATH/../bin/"
echo "================================================"
