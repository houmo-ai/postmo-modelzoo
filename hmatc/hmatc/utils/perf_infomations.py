# Copyright 2025 HOUMO AI
#
# File: perf_infomations.py
# Description:
#   perf_infomations functions
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
import time
from loguru import logger
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from enum import IntEnum, unique

@unique
class PERFTYPE(IntEnum):
    PREFILL_TOTAL_TIME      = 0
    PREFILL_TOKEN_TIME      = 1
    PREFILL_EMBED_TIME      = 2
    PREFILL_INPUT_TIME      = 3
    PREFILL_INFER_TIME      = 4
    PREFILL_OUTPUT_TIME     = 5

    DECODE_TOTAL_TIME       = 6
    DECODE_TOKEN_TIME       = 7
    DECODE_EMBED_TIME       = 8
    DECODE_INPUT_TIME       = 9
    DECODE_INFER_TIME       = 10
    DECODE_OUTPUT_TIME      = 11

    # Vision processing times
    VISION_TOTAL_TIME       = 12
    VISION_PREPROCESS_TIME  = 13
    VISION_INPUT_TIME       = 14
    VISION_INFER_TIME       = 15
    VISION_OUTPUT_TIME      = 16

    PREFILL_LOAD_TIME = 17
    DECODE_LOAD_TIME  = 18
    VISION_LOAD_TIME  = 19

@dataclass
class PerfInformations:
    # Time statistics (ms)
    total_time: float = 0.0
    tokenizer_time: float = 0.0
    embedding_time: float = 0.0
    setinput_time: float = 0.0
    infer_time: float = 0.0
    getoutput_time: float = 0.0

    setinput_time_per_token: float = 0.0
    infer_time_per_token: float = 0.0
    getoutput_time_per_token: float = 0.0

    # Speed (tokens/s)
    total_speed: float = 0.0
    tokenizer_speed: float = 0.0
    embedding_speed: float = 0.0
    setinput_speed: float = 0.0
    infer_speed: float = 0.0
    getoutput_speed: float = 0.0

    # Time start and time end (s)
    total_time_start: float = 0.0
    tokenizer_time_start: float = 0.0
    embedding_time_start: float = 0.0
    setinput_time_start: float = 0.0
    infer_time_start: float = 0.0
    getoutput_time_start: float = 0.0

    # Vision processing times (ms)
    vision_total_time: float = 0.0
    vision_preprocess_time: float = 0.0
    vision_setinput_time: float = 0.0
    vision_infer_time: float = 0.0
    vision_getoutput_time: float = 0.0

    # Vision speed (images/s or items/s)
    vision_total_speed: float = 0.0
    vision_preprocess_speed: float = 0.0
    vision_infer_speed: float = 0.0

@dataclass
class InferenceMetrics:
    """Inference performance metrics data class: unified storage for all performance metrics"""
    # Basic configuration
    batch_size: int = 0
    input_seq_length: int = 0
    output_seq_length: int = 0
    num_images: int = 0  # Number of images processed in vision stage

    # Performance info
    prefill_perf_infos: PerfInformations = field(default_factory=PerfInformations)
    decode_perf_infos: PerfInformations = field(default_factory=PerfInformations)
    vision_perf_infos: PerfInformations = field(default_factory=PerfInformations)  # Vision performance info

    # Summary metrics
    ttft: float = 0.0             # First token time (ms)
    tpot: float = 0.0             # Time per output token (ms/token)
    e2e_time: float = 0.0
    e2e_tps: float = 0.0          # End-to-end token throughput (tokens/s)
    prefill_load_time: float = 0.0
    prefill_load_time_start: float = 0.0
    decode_load_time: float = 0.0
    decode_load_time_start: float = 0.0
    vision_load_time: float = 0.0
    vision_load_time_start: float = 0.0

class InferencePerformanceTracker:
    """
    Inference performance tracker: Unified collection, calculation, and output of inference performance metrics
    Core functions:
    1. Record Prefill/Decode/Vision phase time consumption and token/image counts
    2. Automatically calculate all derived performance metrics (speed, latency, throughput)
    3. Support detailed logging and data export
    4. Provide single/multiple inference metric summaries
    """
    def __init__(self):
        # Initialize metric storage
        self.current_metrics = InferenceMetrics()
        self.current_metrics.e2e_time = time.time()

        # Mapping for perf_start operations
        self.start_time_mapping = {
            PERFTYPE.PREFILL_TOTAL_TIME: lambda: setattr(self.current_metrics.prefill_perf_infos, 'total_time_start', time.time()),
            PERFTYPE.PREFILL_TOKEN_TIME: lambda: setattr(self.current_metrics.prefill_perf_infos, 'tokenizer_time_start', time.time()),
            PERFTYPE.PREFILL_EMBED_TIME: lambda: setattr(self.current_metrics.prefill_perf_infos, 'embedding_time_start', time.time()),
            PERFTYPE.PREFILL_INPUT_TIME: lambda: setattr(self.current_metrics.prefill_perf_infos, 'setinput_time_start', time.time()),
            PERFTYPE.PREFILL_INFER_TIME: lambda: setattr(self.current_metrics.prefill_perf_infos, 'infer_time_start', time.time()),
            PERFTYPE.PREFILL_OUTPUT_TIME: lambda: setattr(self.current_metrics.prefill_perf_infos, 'getoutput_time_start', time.time()),
            PERFTYPE.DECODE_TOTAL_TIME: lambda: setattr(self.current_metrics.decode_perf_infos, 'total_time_start', time.time()),
            PERFTYPE.DECODE_TOKEN_TIME: lambda: setattr(self.current_metrics.decode_perf_infos, 'tokenizer_time_start', time.time()),
            PERFTYPE.DECODE_EMBED_TIME: lambda: setattr(self.current_metrics.decode_perf_infos, 'embedding_time_start', time.time()),
            PERFTYPE.DECODE_INPUT_TIME: lambda: setattr(self.current_metrics.decode_perf_infos, 'setinput_time_start', time.time()),
            PERFTYPE.DECODE_INFER_TIME: lambda: setattr(self.current_metrics.decode_perf_infos, 'infer_time_start', time.time()),
            PERFTYPE.DECODE_OUTPUT_TIME: lambda: setattr(self.current_metrics.decode_perf_infos, 'getoutput_time_start', time.time()),
            PERFTYPE.VISION_TOTAL_TIME: lambda: setattr(self.current_metrics.vision_perf_infos, 'total_time_start', time.time()),
            PERFTYPE.VISION_PREPROCESS_TIME: lambda: setattr(self.current_metrics.vision_perf_infos, 'vision_preprocess_time_start', time.time()),
            PERFTYPE.VISION_INPUT_TIME: lambda: setattr(self.current_metrics.vision_perf_infos, 'setinput_time_start', time.time()),
            PERFTYPE.VISION_INFER_TIME: lambda: setattr(self.current_metrics.vision_perf_infos, 'infer_time_start', time.time()),
            PERFTYPE.VISION_OUTPUT_TIME: lambda: setattr(self.current_metrics.vision_perf_infos, 'getoutput_time_start', time.time()),
            PERFTYPE.PREFILL_LOAD_TIME: lambda: setattr(self.current_metrics, 'prefill_load_time_start', time.time()),
            PERFTYPE.DECODE_LOAD_TIME: lambda: setattr(self.current_metrics, 'decode_load_time_start', time.time()),
            PERFTYPE.VISION_LOAD_TIME: lambda: setattr(self.current_metrics, 'vision_load_time_start', time.time())
        }

        # Mapping for perf_end operations
        self.end_time_mapping = {
            PERFTYPE.PREFILL_TOTAL_TIME: lambda time_diff_ms: setattr(self.current_metrics.prefill_perf_infos, 'total_time', getattr(self.current_metrics.prefill_perf_infos, 'total_time') + time_diff_ms),
            PERFTYPE.PREFILL_TOKEN_TIME: lambda time_diff_ms: setattr(self.current_metrics.prefill_perf_infos, 'tokenizer_time', getattr(self.current_metrics.prefill_perf_infos, 'tokenizer_time') + time_diff_ms),
            PERFTYPE.PREFILL_EMBED_TIME: lambda time_diff_ms: setattr(self.current_metrics.prefill_perf_infos, 'embedding_time', getattr(self.current_metrics.prefill_perf_infos, 'embedding_time') + time_diff_ms),
            PERFTYPE.PREFILL_INPUT_TIME: lambda time_diff_ms: setattr(self.current_metrics.prefill_perf_infos, 'setinput_time', getattr(self.current_metrics.prefill_perf_infos, 'setinput_time') + time_diff_ms),
            PERFTYPE.PREFILL_INFER_TIME: lambda time_diff_ms: setattr(self.current_metrics.prefill_perf_infos, 'infer_time', getattr(self.current_metrics.prefill_perf_infos, 'infer_time') + time_diff_ms),
            PERFTYPE.PREFILL_OUTPUT_TIME: lambda time_diff_ms: setattr(self.current_metrics.prefill_perf_infos, 'getoutput_time', getattr(self.current_metrics.prefill_perf_infos, 'getoutput_time') + time_diff_ms),
            PERFTYPE.DECODE_TOTAL_TIME: lambda time_diff_ms: setattr(self.current_metrics.decode_perf_infos, 'total_time', getattr(self.current_metrics.decode_perf_infos, 'total_time') + time_diff_ms),
            PERFTYPE.DECODE_TOKEN_TIME: lambda time_diff_ms: setattr(self.current_metrics.decode_perf_infos, 'tokenizer_time', getattr(self.current_metrics.decode_perf_infos, 'tokenizer_time') + time_diff_ms),
            PERFTYPE.DECODE_EMBED_TIME: lambda time_diff_ms: setattr(self.current_metrics.decode_perf_infos, 'embedding_time', getattr(self.current_metrics.decode_perf_infos, 'embedding_time') + time_diff_ms),
            PERFTYPE.DECODE_INPUT_TIME: lambda time_diff_ms: setattr(self.current_metrics.decode_perf_infos, 'setinput_time', getattr(self.current_metrics.decode_perf_infos, 'setinput_time') + time_diff_ms),
            PERFTYPE.DECODE_INFER_TIME: lambda time_diff_ms: setattr(self.current_metrics.decode_perf_infos, 'infer_time', getattr(self.current_metrics.decode_perf_infos, 'infer_time') + time_diff_ms),
            PERFTYPE.DECODE_OUTPUT_TIME: lambda time_diff_ms: setattr(self.current_metrics.decode_perf_infos, 'getoutput_time', getattr(self.current_metrics.decode_perf_infos, 'getoutput_time') + time_diff_ms),
            PERFTYPE.VISION_TOTAL_TIME: lambda time_diff_ms: setattr(self.current_metrics.vision_perf_infos, 'vision_total_time', getattr(self.current_metrics.vision_perf_infos, 'vision_total_time') + time_diff_ms),
            PERFTYPE.VISION_PREPROCESS_TIME: lambda time_diff_ms: setattr(self.current_metrics.vision_perf_infos, 'vision_preprocess_time', getattr(self.current_metrics.vision_perf_infos, 'vision_preprocess_time') + time_diff_ms),
            PERFTYPE.VISION_INPUT_TIME: lambda time_diff_ms: setattr(self.current_metrics.vision_perf_infos, 'setinput_time', getattr(self.current_metrics.vision_perf_infos, 'setinput_time') + time_diff_ms),
            PERFTYPE.VISION_INFER_TIME: lambda time_diff_ms: setattr(self.current_metrics.vision_perf_infos, 'infer_time', getattr(self.current_metrics.vision_perf_infos, 'infer_time') + time_diff_ms),
            PERFTYPE.VISION_OUTPUT_TIME: lambda time_diff_ms: setattr(self.current_metrics.vision_perf_infos, 'getoutput_time', getattr(self.current_metrics.vision_perf_infos, 'getoutput_time') + time_diff_ms),
            PERFTYPE.PREFILL_LOAD_TIME: lambda time_diff_ms: setattr(self.current_metrics, 'prefill_load_time', getattr(self.current_metrics, 'prefill_load_time') + time_diff_ms),
            PERFTYPE.DECODE_LOAD_TIME: lambda time_diff_ms: setattr(self.current_metrics, 'decode_load_time', getattr(self.current_metrics, 'decode_load_time') + time_diff_ms),
            PERFTYPE.VISION_LOAD_TIME: lambda time_diff_ms: setattr(self.current_metrics, 'vision_load_time', getattr(self.current_metrics, 'vision_load_time') + time_diff_ms)
        }

        # Mapping for getting start times
        self.get_start_time_mapping = {
            PERFTYPE.PREFILL_TOTAL_TIME: lambda: self.current_metrics.prefill_perf_infos.total_time_start,
            PERFTYPE.PREFILL_TOKEN_TIME: lambda: self.current_metrics.prefill_perf_infos.tokenizer_time_start,
            PERFTYPE.PREFILL_EMBED_TIME: lambda: self.current_metrics.prefill_perf_infos.embedding_time_start,
            PERFTYPE.PREFILL_INPUT_TIME: lambda: self.current_metrics.prefill_perf_infos.setinput_time_start,
            PERFTYPE.PREFILL_INFER_TIME: lambda: self.current_metrics.prefill_perf_infos.infer_time_start,
            PERFTYPE.PREFILL_OUTPUT_TIME: lambda: self.current_metrics.prefill_perf_infos.getoutput_time_start,
            PERFTYPE.DECODE_TOTAL_TIME: lambda: self.current_metrics.decode_perf_infos.total_time_start,
            PERFTYPE.DECODE_TOKEN_TIME: lambda: self.current_metrics.decode_perf_infos.tokenizer_time_start,
            PERFTYPE.DECODE_EMBED_TIME: lambda: self.current_metrics.decode_perf_infos.embedding_time_start,
            PERFTYPE.DECODE_INPUT_TIME: lambda: self.current_metrics.decode_perf_infos.setinput_time_start,
            PERFTYPE.DECODE_INFER_TIME: lambda: self.current_metrics.decode_perf_infos.infer_time_start,
            PERFTYPE.DECODE_OUTPUT_TIME: lambda: self.current_metrics.decode_perf_infos.getoutput_time_start,
            PERFTYPE.VISION_TOTAL_TIME: lambda: self.current_metrics.vision_perf_infos.total_time_start,
            PERFTYPE.VISION_PREPROCESS_TIME: lambda: self.current_metrics.vision_perf_infos.vision_preprocess_time_start,
            PERFTYPE.VISION_INPUT_TIME: lambda: self.current_metrics.vision_perf_infos.setinput_time_start,
            PERFTYPE.VISION_INFER_TIME: lambda: self.current_metrics.vision_perf_infos.infer_time_start,
            PERFTYPE.VISION_OUTPUT_TIME: lambda: self.current_metrics.vision_perf_infos.getoutput_time_start,
            PERFTYPE.PREFILL_LOAD_TIME: lambda: self.current_metrics.prefill_load_time_start,
            PERFTYPE.DECODE_LOAD_TIME: lambda: self.current_metrics.decode_load_time_start,
            PERFTYPE.VISION_LOAD_TIME: lambda: self.current_metrics.vision_load_time_start
        }

    def perf_start(self, perf_type: PERFTYPE):
        """Start timing for specified performance type using dictionary mapping"""
        if perf_type in self.start_time_mapping:
            self.start_time_mapping[perf_type]()
        else:
            raise ValueError(f"Invalid perf_type: {perf_type}")

    def perf_end(self, perf_type: PERFTYPE):
        """End timing for specified performance type and accumulate time using dictionary mapping"""
        if perf_type not in self.get_start_time_mapping:
            raise ValueError(f"Invalid perf_type: {perf_type}")

        current_time = time.time()
        time_diff_ms = (current_time - self.get_start_time_mapping[perf_type]()) * 1000

        if perf_type in self.end_time_mapping:
            self.end_time_mapping[perf_type](time_diff_ms)
        else:
            raise ValueError(f"Invalid perf_type: {perf_type}")

    def calculate_prefill(self, metrics):
        if metrics.prefill_perf_infos.tokenizer_time > 0:
            metrics.prefill_perf_infos.tokenizer_speed = metrics.input_seq_length / (metrics.prefill_perf_infos.tokenizer_time / 1000)

        if metrics.prefill_perf_infos.embedding_time > 0:
            metrics.prefill_perf_infos.embedding_speed = metrics.input_seq_length / (metrics.prefill_perf_infos.embedding_time / 1000)

        if metrics.prefill_perf_infos.setinput_time > 0:
            metrics.prefill_perf_infos.setinput_speed = metrics.input_seq_length / (metrics.prefill_perf_infos.setinput_time / 1000)

        if metrics.prefill_perf_infos.infer_time > 0:
            metrics.prefill_perf_infos.infer_speed = metrics.batch_size * metrics.input_seq_length / (metrics.prefill_perf_infos.infer_time / 1000)

        if metrics.prefill_perf_infos.getoutput_time > 0:
            metrics.prefill_perf_infos.getoutput_speed = metrics.input_seq_length / (metrics.prefill_perf_infos.getoutput_time / 1000)

        if metrics.prefill_perf_infos.total_time > 0:
            metrics.prefill_perf_infos.total_speed = metrics.batch_size * metrics.input_seq_length / (metrics.prefill_perf_infos.total_time / 1000)

    def calculate_decode(self, metrics):
        if metrics.decode_perf_infos.tokenizer_time > 0:
            metrics.decode_perf_infos.tokenizer_speed = (metrics.output_seq_length * metrics.batch_size) / (metrics.decode_perf_infos.tokenizer_time / 1000)

        if metrics.decode_perf_infos.embedding_time > 0:
            metrics.decode_perf_infos.embedding_speed = (metrics.output_seq_length * metrics.batch_size) / (metrics.decode_perf_infos.embedding_time / 1000)

        if metrics.decode_perf_infos.setinput_time > 0:
            metrics.decode_perf_infos.setinput_speed = (metrics.output_seq_length * metrics.batch_size) / (metrics.decode_perf_infos.setinput_time / 1000)
            metrics.decode_perf_infos.setinput_time_per_token = metrics.decode_perf_infos.setinput_time / (metrics.output_seq_length * metrics.batch_size)

        if metrics.decode_perf_infos.infer_time > 0:
            metrics.decode_perf_infos.infer_speed = (metrics.output_seq_length * metrics.batch_size) / (metrics.decode_perf_infos.infer_time / 1000)
            metrics.decode_perf_infos.infer_time_per_token = metrics.decode_perf_infos.infer_time / (metrics.output_seq_length * metrics.batch_size)

        if metrics.decode_perf_infos.getoutput_time > 0:
            metrics.decode_perf_infos.getoutput_speed = (metrics.output_seq_length * metrics.batch_size) / (metrics.decode_perf_infos.getoutput_time / 1000)
            metrics.decode_perf_infos.getoutput_time_per_token = metrics.decode_perf_infos.getoutput_time / (metrics.output_seq_length * metrics.batch_size)

        if metrics.decode_perf_infos.total_time > 0:
            metrics.decode_perf_infos.total_speed = (metrics.output_seq_length * metrics.batch_size) / (metrics.decode_perf_infos.total_time / 1000)

    def calculate_metrics(self):
        """Calculate derived metrics (speed, TTFT, TPOT, TPS)"""
        metrics = self.current_metrics

        # 1. Calculate Prefill stage speeds (tokens/s)
        self.calculate_prefill(metrics)

        # 2. Calculate Decode stage speeds (tokens/s)
        self.calculate_decode(metrics)

        # 3. Calculate Vision stage speeds (images/s or items/s)
        if metrics.vision_perf_infos.vision_preprocess_time > 0 and metrics.num_images > 0:
            metrics.vision_perf_infos.vision_preprocess_speed = metrics.num_images / (metrics.vision_perf_infos.vision_preprocess_time / 1000)
        if metrics.vision_perf_infos.infer_time > 0 and metrics.num_images > 0:
            metrics.vision_perf_infos.vision_infer_speed = metrics.num_images / (metrics.vision_perf_infos.infer_time / 1000)
        if metrics.vision_perf_infos.vision_total_time > 0 and metrics.num_images > 0:
            metrics.vision_perf_infos.vision_total_speed = metrics.num_images / (metrics.vision_perf_infos.vision_total_time / 1000)

        # 4. Calculate summary metrics
        metrics.ttft = metrics.prefill_perf_infos.total_time
        if metrics.output_seq_length > 0:
            metrics.tpot = metrics.decode_perf_infos.total_time / (metrics.output_seq_length * metrics.batch_size)  # TPOT = Decode total time / number of output tokens

        # E2E TPS = total processed tokens / total time (tokens/s)
        total_tokens =  (metrics.output_seq_length + 1)
        e2e_time_s = metrics.e2e_time
        if e2e_time_s > 0:
            metrics.e2e_tps = metrics.batch_size * total_tokens / e2e_time_s

    def set_basic_info(self, batch_size: int, input_seq_length: int, output_seq_length: int, num_images: int = 0):
        """Set basic inference configuration"""
        self.current_metrics.batch_size = batch_size
        self.current_metrics.input_seq_length = input_seq_length
        self.current_metrics.output_seq_length = output_seq_length
        self.current_metrics.num_images = num_images
        self.current_metrics.e2e_time = time.time() - self.current_metrics.e2e_time

    def reset_perf_time(self):
        self.current_metrics.e2e_time = time.time()
        for perf_type in self.start_time_mapping:
            if perf_type < PERFTYPE.PREFILL_LOAD_TIME:
                self.start_time_mapping[perf_type]()

    def show_summary(self) -> None:
        """Print formatted performance summary report"""
        self.calculate_metrics()
        metrics = self.current_metrics

        logger.success("=" * 100)
        logger.success("                    Model Inference Performance Summary Report")
        logger.success("=" * 100)

        # Basic Configuration
        logger.success(f"Configuration Details:")
        logger.success(f"  Batch Size: {metrics.batch_size:>6}")
        logger.success(f"  Input Length per Sample: {metrics.input_seq_length:>6} tokens")
        logger.success(f"  Output Length per Sample: {(metrics.output_seq_length + 1):>6} tokens")
        if metrics.num_images > 0:
            logger.success(f"  Number of Images: {metrics.num_images:>6} images")

        if metrics.prefill_load_time > 0:
            logger.success(f"  Prefill Model Load Time: {metrics.prefill_load_time:>7.2f}ms")
        if metrics.decode_load_time > 0:
            logger.success(f"  Decode Model Load Time: {metrics.decode_load_time:>7.2f}ms")
        if metrics.vision_load_time > 0:
            logger.success(f"  Vision Model Load Time: {metrics.vision_load_time:>7.2f}ms")

        # Vision Stage Performance (if applicable)
        if metrics.num_images > 0 and (metrics.vision_perf_infos.vision_total_time > 0 or
                                       metrics.vision_perf_infos.vision_preprocess_time > 0):
            logger.success(f"Vision Stage Performance:")
            logger.success(f"  Total Time: {metrics.vision_perf_infos.vision_total_time:>7.2f}ms | Speed: {metrics.vision_perf_infos.vision_total_speed:>7.2f} images/s")
            logger.success(f"  Preprocessing Time: {metrics.vision_perf_infos.vision_preprocess_time:>5.2f}ms | Speed: {metrics.vision_perf_infos.vision_preprocess_speed:>7.2f} images/s")
            logger.success(f"  API SetInput Time: {metrics.vision_perf_infos.setinput_time:>6.2f}ms")
            logger.success(f"  API Inference Time: {metrics.vision_perf_infos.infer_time:>5.2f}ms | Speed: {metrics.vision_perf_infos.vision_infer_speed:>7.2f} images/s")
            logger.success(f"  API GetOutput Time: {metrics.vision_perf_infos.getoutput_time:>5.2f}ms")

        # Prefill Stage Performance
        logger.success(f"Prefill Stage Performance:")
        logger.success(f"  Total Time: {metrics.prefill_perf_infos.total_time:>7.2f}ms | Speed: {metrics.prefill_perf_infos.total_speed:>7.2f} tokens/s")
        if metrics.prefill_perf_infos.tokenizer_time > 0:
            logger.success(f"  Tokenization Time: {metrics.prefill_perf_infos.tokenizer_time:>7.2f}ms")
        else:
            logger.success(f"  Tokenization Time: Skipped (No operation)")
        logger.success(f"  Embedding Time: {metrics.prefill_perf_infos.embedding_time:>7.2f}ms")
        logger.success(f"  API SetInput Time: {metrics.prefill_perf_infos.setinput_time:>6.2f}ms")
        logger.success(f"  API Inference Time: {metrics.prefill_perf_infos.infer_time:>5.2f}ms | Prefill Speed: {metrics.prefill_perf_infos.infer_speed:>7.2f} tokens/s")
        logger.success(f"  API GetOutput Time: {metrics.prefill_perf_infos.getoutput_time:>5.2f}ms")

        # Decode Stage Performance
        logger.success(f"Decode Stage Performance:")
        logger.success(f"  Total Time: {metrics.decode_perf_infos.total_time:>7.2f}ms | Speed: {metrics.decode_perf_infos.total_speed:>7.2f} tokens/s")
        if metrics.decode_perf_infos.tokenizer_time > 0:
            logger.success(f"  Tokenization Time: {metrics.decode_perf_infos.tokenizer_time:>7.2f}ms")
        else:
            logger.success(f"  Tokenization Time: Skipped (No operation)")
        logger.success(f"  Embedding Time: {metrics.decode_perf_infos.embedding_time:>7.2f}ms")
        logger.success(f"  API SetInput Time: {metrics.decode_perf_infos.setinput_time_per_token:>6.2f}ms")
        logger.success(f"  API Inference Time: {metrics.decode_perf_infos.infer_time_per_token:>5.2f}ms | Decode Speed: {metrics.decode_perf_infos.infer_speed:>7.2f} tokens/s")
        logger.success(f"  API GetOutput Time: {metrics.decode_perf_infos.getoutput_time_per_token:>5.2f}ms")

        # Summary Metrics
        logger.success(f"Overall Performance Metrics:")
        logger.success(f"  TTFT (Time To First Token): {metrics.ttft:>7.2f} ms")
        logger.success(f"  TPOT (Time Per Output Token): {metrics.tpot:>5.2f} ms/token")
        logger.success(f"  E2E Latency (End-to-End): {metrics.e2e_time:>9.2f} seconds")
        logger.success(f"  E2E TPS (Throughput): {metrics.e2e_tps:>13.2f} tokens/s")

        logger.success("=" * 100)