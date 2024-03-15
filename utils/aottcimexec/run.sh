#!/usr/bin/env bash

TEST_COUNT=1000
TEST_MODEL=$MODELZOO_PATH/models/backbone/resnet50/tcim_resnet50
#TEST_DATA=$MODELZOO_PATH/cv/resnet50/inference_model/preprocessed
TEST_LABEL=$DATASETS_PATH/ILSVRC2012_val_labels.txt
TEST_WARMUP=10
DRIVER_PATH="$1"

./tcimexec --model $TEST_MODEL --iterations $TEST_COUNT --warm_up $TEST_WARMUP
