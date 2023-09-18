#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

CPU_COUNT=$(grep 'processor' /proc/cpuinfo | sort | uniq | wc -l)
if [ "${CPU_COUNT}" == "0" ]; then
  CPU_COUNT=1
fi
if [ -z "${IS_DEBUG}" ]; then
  IMAGE_COUNT=5000
else
  IMAGE_COUNT=5
fi
if [ -d preprocessed ]; then
  PREPROCESSED=$(find ./preprocessed -name "*.jpg" | wc -l)
else
  PREPROCESSED=0
fi

COCO_PATH="${DATASETS_PATH}/coco2017"

if [[ ! -f "${COCO_PATH}" ]];then
  COCO_PATH="${DATASETS_PATH}/COCO"
fi

if [ ${PREPROCESSED} -lt ${IMAGE_COUNT} ]; then
  python3 preprocess.py --output-path ./preprocessed \
          --coco-path "${COCO_PATH}" \
          --count ${IMAGE_COUNT} \
          -n "${CPU_COUNT}"
fi

if [ -c /dev/hmcl_feature_in ]; then
  export HDPL_PLATFORM=ASIC
fi

mkdir -p build
cd build
cmake ..
make
cd ../
./build/hdpl_yolov3 "${COCO_PATH}/annotations/instances_val2017.json" ./preprocessed ${IMAGE_COUNT}
if pip3 show pycocotools; then
  ./cal_meanap.py --predict-result ./detections.json --coco-path "${COCO_PATH}"
fi
