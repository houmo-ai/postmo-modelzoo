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
mode_file_name="vit_small.onnx"
if [ -f "${mode_file_name}" ];
then
  echo "${mode_file_name} already exists."
else
  echo "Downloading ${mode_file_name} file"
  wget -q -O ${mode_file_name} http://10.10.1.53:8081/artifactory/model_zoo2/houmo/vit/vit_small_patch16_224.onnx
fi

if [ ! -f "${DATASETS_PATH}/imagenet/ILSVRC2012_val_00000001.JPEG" ];
then
    echo "Please download LSVRC_2012_img_val datasets from https://image-net.org/challenges/LSVRC/ to ${DATASETS_PATH}/imagenet"
    exit 1
fi
