#!/usr/bin/env bash

TEST_COUNT=10
TEST_MODEL=$MODELZOO_PATH/cv/resnet50/compile_model/resnet50
TEST_DATA=$MODELZOO_PATH/cv/resnet50/inference_model/preprocessed
TEST_LABEL=$DATASETS_PATH/ILSVRC2012_val_labels.txt
TEST_WARMUP=10

./multi_stream_tcimexec --model $TEST_MODEL --iterations $TEST_COUNT --warm_up $TEST_WARMUP
