#!/usr/bin/env bash

set -e

model_case=(
    $MODELZOO_PATH/models/backbone/resnet50
    $MODELZOO_PATH/models/backbone/mobilenetv2
    $MODELZOO_PATH/models/backbone/efficientnet
    $MODELZOO_PATH/models/detection/yolov3
    $MODELZOO_PATH/models/detection/yolov5s
    $MODELZOO_PATH/models/autodrive/yolop
)

for case in "${model_case[@]}"
do
    echo "===> $case test begin"
    cd $case
    rm -rf output/H30/result
    python3 get_model.py --type quant
    hmbuild.sh
    echo "===> $case test end"
done