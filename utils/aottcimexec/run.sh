#!/usr/bin/env bash

TEST_COUNT=1000
TEST_MODEL=$MODELZOO_PATH/tests/resnet50_32x32_cifar10/tcim_resnet50_32x32_cifar10
#TEST_DATA=$MODELZOO_PATH/cv/resnet50/inference_model/preprocessed
TEST_LABEL=$DATASETS_PATH/ILSVRC2012_val_labels.txt
TEST_WARMUP=10
DRIVER_PATH="$1"

cat << EOF
if [ X"${DRIVER_PATH}" == X ]; then
  DRIVER_PATH="/home/debug.sw/0.9.8.20231206/"
fi

cd ${DRIVER_PATH}
./reset_aicore.sh
cd -
EOF

./tcimexec --model $TEST_MODEL --iterations $TEST_COUNT --warm_up $TEST_WARMUP
