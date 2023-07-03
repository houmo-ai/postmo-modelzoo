#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

CPU_COUNT=$(grep 'physical id' /proc/cpuinfo | sort | uniq | wc -l)
if [ -z "${IS_DEBUG}" ]; then
  IMAGE_COUNT=50000
else
  IMAGE_COUNT=10
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

cd ../../../utils/classification
bash build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -
hdpl_classification --model ../compile_model/mobilenet_v2 --label "${DATASETS_PATH}/ILSVRC2012_val_labels.txt" --data_root ./preprocessed --count ${IMAGE_COUNT}
