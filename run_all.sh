#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"

# set modelzoo env
export TCIM_INC_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/include
export TCIM_LIB_PATH=$HDPL_TOOLCHAIN_ITVM_INSTALL/lib
export HDPL_INC_PATH=$HDPL_PATH/include
export HDPL_LIB_PATH=$HDPL_PATH/lib
export IDNNL_INC_PATH=$IDNNL_PATH/include
export IDNNL_LIB_PATH=$IDNNL_PATH/lib

echo "HDPL_TOOLCHAIN_ITVM_INSTALL is $HDPL_TOOLCHAIN_ITVM_INSTALL"
echo "HDPL_PATH is $HDPL_PATH"

export CMAKE_CONFIG_PATH=$SCRIPT_PATH/develop.cmake

if [[ -z "${DATASETS_PATH}" ]]; then
  export DATASETS_PATH=$SCRIPT_PATH/data/datasets
fi
if [[ -z "${MODEL_PATH}" ]]; then
  export MODEL_PATH=$SCRIPT_PATH/data/models
fi

# For bash 4.4+, must not be in posix mode, may use temporary files
# perf_scripts=()
# while IFS='' read -r line; do perf_scripts+=("$line"); done < <(find . -type f -name "perf.sh")
# for script in "${perf_scripts[@]}"; do
#   echo "Run ${script}"
  # bash "${script}"
# done

# For bash 4.4+, must not be in posix mode, may use temporary files
# eval_scripts=()
# while IFS='' read -r line; do eval_scripts+=("$line"); done < <(find . -type f -name "eval.sh")
# for script in "${eval_scripts[@]}"; do
#   echo "Run ${script}"
  # bash "${script}"
# done

bash ./cv/yolov3/perf.sh
bash ./cv/mobilenet_v2/perf.sh
bash ./cv/yolop/perf.sh
bash ./cv/resnet50/perf.sh
bash ./cv/point_pillars/perf.sh
bash ./cv/efficientnet/perf.sh
bash ./tests/tracking/perf.sh
bash ./tests/resizer/perf.sh
bash ./tests/traffic_light/perf.sh
bash ./tests/fastbev2d/perf.sh
# bash ./tests/resnet50_576_1024/perf.sh
bash ./tests/yolov5s/perf.sh
bash ./tests/fastbev3d/perf.sh
bash ./tests/resnet50/perf.sh
bash ./tests/yolop_576_1024/perf.sh
bash ./cv/yolov3/eval.sh
bash ./cv/mobilenet_v2/eval.sh
# bash ./cv/resnet50/eval.sh
bash ./cv/efficientnet/eval.sh
# bash ./tests/resnet50/eval.sh
