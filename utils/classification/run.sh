#!/usr/bin/env bash

TEST_COUNT=10
TEST_MODEL=$MODELZOO_PATH/cv/resnet50/compile_model/resnet50
TEST_DATA=$MODELZOO_PATH/cv/resnet50/inference_model/preprocessed
TEST_LABEL=$DATASETS_PATH/ILSVRC2012_val_labels.txt

./hdpl_classification --model $TEST_MODEL --label $TEST_LABEL --data_root $TEST_DATA --count $TEST_COUNT
