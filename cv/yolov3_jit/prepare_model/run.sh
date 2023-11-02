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
model_file_name="quant_yolov3.onnx"
if [ -f "${model_file_name}" ];
then
  echo "${model_file_name} already exists."
else
  echo "Downloading ${model_file_name} file"
  wget -q -O ${model_file_name} http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/hmquant_yolov3_yuv.onnx
fi
