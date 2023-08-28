#!/usr/bin/env bash

TEST_COUNT=10
TEST_MODEL=$MODELZOO_PATH/cv/resnet50/compile_model/resnet50
TEST_DATA=$MODELZOO_PATH/cv/resnet50/inference_model/preprocessed
TEST_LABEL=$DATASETS_PATH/ILSVRC2012_val_labels.txt
TEST_WARMUP=10
TEST_THREAD=4
TEST_STREAM=4

./multi_thread_stream_tcimexec -m $TEST_MODEL -i $TEST_COUNT -w $TEST_WARMUP -t $TEST_THREAD -s $TEST_STREAM
