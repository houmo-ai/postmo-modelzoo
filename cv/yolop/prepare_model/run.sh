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
if [ -f "yolop.onnx" ];
then
  echo "yolop.onnx already exists."
else
  echo "Downloading yolop.onnx file"
  wget -q -O yolop.onnx http://10.10.1.53:8082/artifactory/model_zoo2/yolop/yolop-192-320-without_postprocess.onnx
fi
