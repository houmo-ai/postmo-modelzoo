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
if [ -f "yolov3.onnx" ];
then
  echo "yolov3.onnx already exists."
else
  echo "Downloading yolov3.onnx file"
  wget -q -O yolov3.onnx http://10.10.1.53:8082/artifactory/model_zoo/yolov3/yolov3.onnx
fi
