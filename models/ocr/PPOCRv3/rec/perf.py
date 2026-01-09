#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: perf.py
# Description:
#   ppocrv3 recognition model inference performance and throughput.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import sys
import os
from hmatc.utils import logger
import hmatc.python.perf as perf

model_path = "./output/xh2/ppocrv3_rec.hmm"
warmup_num = 10
sample_num = 1000
loop_num = 1
device_id = [0]
thread_num = 1
if len(sys.argv) > 1:
    logger.info(f"perf -> hmm path: {sys.argv[1]}")
    model_path = str(sys.argv[1])
elif len(sys.argv) > 2:
    logger.info(f"perf -> warmup_num = {sys.argv[2]}")
    warmup_num = int(sys.argv[2])
elif len(sys.argv) > 3:
    logger.info(f"perf -> sample_num = {sys.argv[3]}")
    sample_num = int(sys.argv[3])
elif len(sys.argv) > 4:
    logger.info(f"perf -> loop_num = {sys.argv[4]}")
    loop_num = int(sys.argv[4])
elif len(sys.argv) > 5:
    logger.info(f"perf -> device_id = {sys.argv[5]}")
    device_id = [int(sys.argv[5])]
elif len(sys.argv) > 6:
    logger.info(f"perf -> thread_num = {sys.argv[6]}")
    thread_num = int(sys.argv[6])

logger.info(f"Using model: {model_path}")
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")

perf_info = perf.CModelRunner(
    model_path,
    warmup_num,
    sample_num,
    loop_num,
    thread_num,
    stream_num=0,
    check_output=False,
    devices=device_id,
)
