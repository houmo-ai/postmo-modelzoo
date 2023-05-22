#!/usr/bin/env bash
set -e

if [ -z "${MODEL_PATH}" ]; then
  echo "Please run env.sh before run the script."
  exit 1
fi

if [ ! -d "${MODEL_PATH}" ]; then
  mkdir -p "${MODEL_PATH}"
fi

cd "${MODEL_PATH}"
if [ -f "resnet50.onnx" ];
then
  echo "resnet50.onnx already exists."
else
  echo "Downloading resnet50.onnx file"
  wget -O resnet50.onnx http://10.10.1.53:8082/artifactory/model_zoo2/houmo/resnet/resnet50_224x224.onnx
fi

if [ ! -f "${DATASETS_PATH}/imagenet/ILSVRC2012_val_00000001.JPEG" ];
then
    echo "Please download LSVRC_2012_img_val datasets from https://image-net.org/challenges/LSVRC/ to ${DATASETS_PATH}/imagenet"
    exit 1
fi
