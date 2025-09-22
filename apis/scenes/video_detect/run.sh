#!/usr/bin/env bash
set -e

# default params
RESIZER_SWITCH=ON
RK_DECODER_SWITCH=OFF
JPEG_SWITCH=OFF

show_help() {
    echo "Usage: $0 [options]"
    echo "  -r, --disable_resizer       disable the resize decoded output function (default: $RESIZER_SWITCH)"
    echo "  -d, --enable_rk_decoder     enable RK decoder and disable HM decoder (default: $RK_DECODER_SWITCH)"
    echo "  -jpeg, --enable_jpeg        enable JPEG encoder&decoder and disable HM decoder (default: $JPEG_SWITCH)"
    echo "  -h, --help                  help information"
    exit 0
}

while [[ $# -gt 0 ]]; do
  case $1 in
    -r|--disable_resizer)
      RESIZER_SWITCH=OFF
      shift 1
      ;;
    -d|--enable_rk_decoder)
      RK_DECODER_SWITCH=ON
      shift 1
      ;;
    -jpeg|--enable_jpeg)
      JPEG_SWITCH=ON
      shift 1
      ;;
    -h|--help)
      show_help
      ;;
    *)
      echo "Error: Unknown parameter '$1'" >&2
      show_help
      ;;
  esac
done

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

if [[ -z $HOUMO_EXAMPLES_PATH ]]; then
  export HOUMO_EXAMPLES_PATH=$WORK_PATH/../../..
fi

# get test model
python3 get_model.py

# c++ example
mkdir -p build
cd build || exit 1

cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH -DENABLE_RESIZER=$RESIZER_SWITCH -DENABLE_RK_DECODER=$RK_DECODER_SWITCH -DENABLE_JPEG=$JPEG_SWITCH -DCMAKE_BUILD_TYPE=Release ..
make -j
make install

if [ "$RK_DECODER_SWITCH" = "ON" ]; then
  export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
fi

cd $WORK_PATH
# ./example_video_detect
