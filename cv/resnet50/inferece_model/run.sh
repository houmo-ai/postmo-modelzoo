#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

cmake .
make

CPU_COUNT=$(grep 'physical id' /proc/cpuinfo | sort | uniq | wc -l)
if [ -z "${IS_DEV}" ]; then
  IMAGE_COUNT=50000
else
  IMAGE_COUNT=50
fi
if [ -d preprocessed ]; then
  PREPROCESSED=$(find ./preprocessed -name "*.JPEG" | wc -l)
else
  PREPROCESSED=0
fi

if [ ${PREPROCESSED} -lt ${IMAGE_COUNT} ]; then
  python3 preprocess.py --output-path ./preprocessed \
          --imagenet-path "${DATASETS_PATH}/imagenet" \
          --count ${IMAGE_COUNT} \
          -n "${CPU_COUNT}"
fi

if [ -c /dev/hmcl_feature_in ]; then
  export HDPL_PLATFORM=ASIC
fi

./hdpl_resnet50_run "${DATASETS_PATH}/ILSVRC2012_val_labels.txt" ./preprocessed ${IMAGE_COUNT}
