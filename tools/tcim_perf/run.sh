#!/usr/bin/env bash

TEST_MODEL=$HOUMO_EXAMPLES_PATH/models/backbone/resnet50/output/xh2/resnet50_xh2_b1_1core_O2.hmm
MODEL_NAME=resnet50_xh2_w8a8h1_sefp
TEST_INPUT=$HOUMO_EXAMPLES_PATH/models/backbone/resnet50/output/xh2/hmquant/golden/step_0
TEST_SAMPLES=1000
TEST_WARMUP=1
TEST_BATCH=1
TEST_THREAD=1

tcim_perf -m $TEST_MODEL -n $MODEL_NAME -i $TEST_INPUT -s $TEST_SAMPLES -w $TEST_WARMUP -b $TEST_BATCH -t $TEST_THREAD
