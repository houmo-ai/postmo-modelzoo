#!/usr/bin/env bash

TEST_COUNT=3
TEST_MODEL=$MODELZOO_PATH/tests/lane/tcim_lane_512x1536
TEST_DATA=$MODELZOO_PATH/cv/resnet50/inference_model/preprocessed
TEST_LABEL=$DATASETS_PATH/ILSVRC2012_val_labels.txt
TEST_WARMUP=10
TEST_THREAD=1
TEST_STREAM=4

./multi_thread_stream_tcimexec -m $TEST_MODEL -i $TEST_COUNT -w $TEST_WARMUP -t $TEST_THREAD -l 1
