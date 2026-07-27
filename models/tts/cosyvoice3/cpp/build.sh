#!/usr/bin/env bash
# Build script for CosyVoice3 C++ demo on Linux

# Parse command line arguments
BUILD_AUDIO_FROM_SOURCE="-DBUILD_AUDIO_FROM_SOURCE=ON"
while [[ $# -gt 0 ]]; do
    case $1 in
        --source)
            BUILD_AUDIO_FROM_SOURCE="-DBUILD_AUDIO_FROM_SOURCE=ON"
            shift
            ;;
        --prebuilt)
            BUILD_AUDIO_FROM_SOURCE=""
            shift
            ;;
        *)
            shift
            ;;
    esac
done

if [ ! -e 3rdparty ];then
  mkdir 3rdparty
fi
if [ ! -e 3rdparty/tokenizers-cpp ];then
  cd ..
  python3 get_model.py --type hmm
  cd cpp
fi
if [ ! -e 3rdparty/audio_3rdparty ];then
  echo "Download precompiled audio libraries."
  python3 scripts/get_3rdparty.py
fi
python3 scripts/convert_embeddings.py
if [ $(uname -s) = "Linux" ]; then
  if [ "$HOUMO_TARGET" = "xh2" ]; then
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH/../bin -DCMAKE_BUILD_TYPE=Release ${BUILD_AUDIO_FROM_SOURCE} ..
    make
    make install
  else
    echo "UnSupport Backend!"
  fi
else
  echo "UnSupport PlatForm!"
fi