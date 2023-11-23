#!/usr/bin/env bash

TEST_LOOP=10
TEST_COUNT=1
TEST_MODEL=$MODELZOO_PATH/cv/resnet50/tcim_resnet50
TEST_DATA=$MODELZOO_PATH/cv/resnet50/output/H30/result/input.1_input.bin
TEST_LABEL=$DATASETS_PATH/ILSVRC2012_val_labels.txt
TEST_WARMUP=10
TEST_THREAD=1
TEST_STREAM=4

./multi_thread_stream_tcimexec -m $TEST_MODEL -d $TEST_DATA -i $TEST_COUNT -w $TEST_WARMUP -t $TEST_THREAD -l $TEST_LOOP
