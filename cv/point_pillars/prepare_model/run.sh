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
if [ -f "pointpillars_pfe.onnx" ];
then
  echo "pointpillars_pfe.onnx already exists."
else
  echo "Downloading pointpillars_pfe.onnx file"
  wget -O pointpillars_pfe.onnx http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/pointpillar/modified_pfe_1.onnx
fi

if [ -f "pointpillars_rpn.onnx" ];
then
  echo "pointpillars_rpn.onnx already exists."
else
  echo "Downloading pointpillars_pfe.onnx file"
  wget -O rpn.tar http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/pointpillar/pointpillars_rpn_464_0423_v2.tar
  tar xf rpn.tar
  mv pointpillars_rpn_464_0423_v2/rpn.onnx pointpillars_rpn.onnx
  rm -rf rpn.tar pointpillars_rpn_464_0423_v2
fi

cd "${SCRIPT_DIR}"
data_file_path=../inference_model/point_464.txt
if [ ! -f ${data_file_path} ]; then
  wget -O ${data_file_path} http://10.10.1.53:8082/artifactory/hdpl_test_data/quant_models/apollo/pointpillar/point_464.txt
fi
