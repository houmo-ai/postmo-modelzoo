#!/usr/bin/env bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

if [ "$#" != "1" ]; then
  echo "Usage: $0 <mode>"
  echo "mode 0: resize 2160x3840 -> 576x1024"
  echo "mode 1: resize 1052x2048 -> 576x1024"
  echo "mode 2: resize 1052x2048 -> 288x512"
  echo "mode 3: crop 576x1024 -> 224x224"
  echo "mode 4: crop 288x512 -> 224x224"
  exit 0
fi

mode=$1

if [ "${mode}" == "0" ]; then
  HEIGHT=2160
  WIDTH=3840
  CROP_HEIGHT=2160
  CROP_WIDTH=3840
  O_HEIGHT=576
  O_WIDTH=1024
  echo "Run mode ${mode} resize: ${CROP_HEIGHT}x${CROP_WIDTH} -> ${O_HEIGHT}x${O_WIDTH}"
elif [ "${mode}" == "1" ]; then
  HEIGHT=1080
  WIDTH=2048
  CROP_HEIGHT=1080
  CROP_WIDTH=2048
  O_HEIGHT=576
  O_WIDTH=1024
  echo "Run mode ${mode} resize: ${CROP_HEIGHT}x${CROP_WIDTH} -> ${O_HEIGHT}x${O_WIDTH}"
elif [ "${mode}" == "2" ]; then
  HEIGHT=1080
  WIDTH=2048
  CROP_HEIGHT=1080
  CROP_WIDTH=2048
  O_HEIGHT=288
  O_WIDTH=512
  echo "Run mode ${mode} resize: ${CROP_HEIGHT}x${CROP_WIDTH} -> ${O_HEIGHT}x${O_WIDTH}"
elif [ "${mode}" == "3" ]; then
  HEIGHT=576
  WIDTH=1024
  CROP_HEIGHT=224
  CROP_WIDTH=224
  O_HEIGHT=224
  O_WIDTH=224
  echo "Run mode ${mode} crop: ${HEIGHT}x${WIDTH} -> ${CROP_HEIGHT}x${CROP_WIDTH}"
elif [ "${mode}" == "4" ]; then
  HEIGHT=288
  WIDTH=512
  CROP_HEIGHT=224
  CROP_WIDTH=224
  O_HEIGHT=244
  O_WIDTH=244
  echo "Run mode ${mode} crop: ${HEIGHT}x${WIDTH} -> ${CROP_HEIGHT}x${CROP_WIDTH}"
else
  echo "Not supported mode"
fi

# compile_model
hdplcc -shared -fPIC -o tcim_resizer.so hdpl_code/lib0.hu hdpl_code/lib1.hu hdpl_code/devc.o -ffunction-sections -fdata-sections  -I/usr/local/lib/python3.8/dist-packages/tvm/include -I/usr/local/lib/python3.8/dist-packages/tvm/include -I/usr/local/houmo/include -DHDPL_ENTRY=__device__ -DIDNNL_ENTRY=__device__ -DHEIGHT=${HEIGHT} -DWIDTH=${WIDTH} -DO_HEIGHT=${O_HEIGHT} -DO_WIDTH=${O_WIDTH} -DCROP_HEIGHT=${CROP_HEIGHT} -DCROP_WIDTH=${CROP_WIDTH}

cd ../../utils/aottcimexec/
./build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -

if [ -c /dev/hm_host_pcie ]; then
  export HDPL_PLATFORM=ASIC
fi

if [ -z "${IS_DEBUG}" ]; then
  ITERATION=1000
else
  ITERATION=1
fi

tcimexec --model "${SCRIPT_DIR}/tcim_resizer" --iterations ${ITERATION}
