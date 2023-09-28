#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

cd "${SCRIPT_DIR}"

CPU_COUNT=$(grep 'processor' /proc/cpuinfo | sort | uniq | wc -l)
if [ "${CPU_COUNT}" == "0" ]; then
  CPU_COUNT=1
fi
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
  OLD_HDPL_PLATFORM=${HDPL_PLATFORM}
  export HDPL_PLATFORM=ISIM
  python3 preprocess.py --output-path ./preprocessed \
          --imagenet-path "${DATASETS_PATH}/imagenet" \
          --count ${IMAGE_COUNT} \
          -n "${CPU_COUNT}"
  export HDPL_PLATFORM=${OLD_HDPL_PLATFORM}
fi


cd ../../../utils/classification
bash build.sh
PATH="$(pwd):${PATH}"
export PATH
cd -
hdpl_classification --model ../compile_model/mobilenet_v2 --label "${DATASETS_PATH}/ILSVRC2012_val_labels.txt" --data_root ./preprocessed --count ${IMAGE_COUNT}
