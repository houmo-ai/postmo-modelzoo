#!/usr/bin/env bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
set -e

if [ -z "${MODEL_PATH}" ]; then
  echo "Please run env.sh before run the script."
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  mkdir -p "${MODEL_PATH}"
fi

cd "${MODEL_PATH}"
if [ -f "efficientnet_b0_224x224.onnx" ];
then
  echo "efficientnet_b0_224x224.onnx already exists."
else
  echo "Downloading efficientnet_b0_224x224.onnx file"
  if [ -z "${IS_DEBUG}" ]; then
    python3 "${SCRIPT_DIR}/gen_onnx_model.py"
  else
    wget http://10.10.1.53:8082/artifactory/model_zoo2/haomo/efficientnet/efficientnet_b0_224x224.onnx
  fi
fi

if [ ! -f "${DATASETS_PATH}/imagenet/ILSVRC2012_val_00000001.JPEG" ];
then
    echo "Please download LSVRC_2012_img_val datasets from https://image-net.org/challenges/LSVRC/ to ${DATASETS_PATH}/imagenet"
    exit 1
fi
