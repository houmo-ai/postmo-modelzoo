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
if [ -f "resnet50_576_1024.onnx" ];
then
  echo "resnet50_576_1024.onnx already exists."
else
  echo "Downloading resnet50_576_1024.onnx file"
  python3 "${SCRIPT_DIR}/gen_onnx_model.py"
fi

if [ ! -f "${DATASETS_PATH}/imagenet/ILSVRC2012_val_00000001.JPEG" ];
then
    echo "Please download LSVRC_2012_img_val datasets from https://image-net.org/challenges/LSVRC/ to ${DATASETS_PATH}/imagenet"
    exit 1
fi
