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
if [ -f "yolop_576_1024.onnx" ];
then
  echo "yolop_576_1024.onnx already exists."
else
  echo "Downloading yolop_576_1024.onnx file"
  wget -q -O yolop_576_1024.onnx http://10.10.1.53:8082/artifactory/model_zoo/yolop/yolop_576x1024_without_yolo.onnx
fi
