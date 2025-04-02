#!/usr/bin/env bash

set -e

model_case=(
    $HOUMO_MODELZOO_PATH/models/backbone/resnet50
    $HOUMO_MODELZOO_PATH/models/backbone/mobilenetv2
    $HOUMO_MODELZOO_PATH/models/backbone/efficientnet
    $HOUMO_MODELZOO_PATH/models/detection/yolov3
    $HOUMO_MODELZOO_PATH/models/detection/yolov5s
    $HOUMO_MODELZOO_PATH/models/autodrive/yolop
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