#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 HOUMO AI
#
# File: qwen_demo_utils.py
# Description:
#   Utility functions for Qwen3 model demos - Contains helper functions for
#   character validation, performance statistics calculation, and logging.
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

from loguru import logger


def is_valid_char(cp) -> bool:
    """
    Check if a Unicode character is a valid Chinese character or English letter.

    Valid character ranges:
    - CJK Unified Ideographs (4E00-9FFF)
    - CJK Unified Ideographs Extension A (3400-4DBF)
    - CJK Unified Ideographs Extension B (20000-2A6DF)
    - CJK Unified Ideographs Extension C (2A700-2B73F)
    - CJK Unified Ideographs Extension D (2B740-2B81F)
    - CJK Unified Ideographs Extension E (2B820-2CEAF)
    - CJK Compatibility Ideographs (F900-FAFF)
    - CJK Compatibility Ideographs Supplement (2F800-2FA1F)
    - English uppercase letters (A-Z)
    - English lowercase letters (a-z)

    Args:
        cp (int): Unicode code point of the character to check.

    Returns:
        bool: True if the character is valid, False otherwise.
    """
    valid_ranges = [
        (0x4E00, 0x9FFF),  # CJK Unified Ideographs
        (0x3400, 0x4DBF),  # CJK Extension A
        (0x20000, 0x2A6DF),  # CJK Extension B
        (0x2A700, 0x2B73F),  # CJK Extension C
        (0x2B740, 0x2B81F),  # CJK Extension D
        (0x2B820, 0x2CEAF),  # CJK Extension E
        (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
        (0x2F800, 0x2FA1F),  # CJK Compatibility Ideographs Supplement
        (0x0041, 0x005A),  # English uppercase letters
        (0x0061, 0x007A),  # English lowercase letters
    ]
    for start, end in valid_ranges:
        if start <= cp <= end:
            return True

    return False


def show_statistics(
    input_tokens, output_tokens, ttft_time, prefill_time, decode_time, total_time
):
    """
    Display comprehensive performance statistics for model inference.

    Args:
        input_tokens (int): Number of input tokens processed during prefill
        output_tokens (int): Total number of output tokens generated
        ttft_time (float): Time to first token in seconds
        prefill_time (float): Time taken for prefill operation in seconds
        decode_time (float): Time taken for decode operations in seconds
        total_time (float): Total end-to-end processing time in seconds
    """
    logger.success(
        f"Total Input: {input_tokens} tokens, Output {output_tokens} tokens, Prefill Cost {prefill_time*1000:.3f} ms, Decode Cost {decode_time*1000:.3f} ms"
    )
    # Prevent division by zero by setting minimum time values
    prefill_time = 0.00001 if prefill_time == 0 else prefill_time
    decode_time = 0.00001 if decode_time == 0 else decode_time
    # Calculate and log token processing speeds
    logger.success(
        f"Prefill Speed: {input_tokens / prefill_time:.2f} tokens/s; Decode Speed: {(output_tokens - 1) / decode_time:.2f} tokens/s"
    )
    logger.success(f"TTFT (Time to First Token): {ttft_time * 1000:.3f} ms")
    logger.success(
        f"TPOT (Time Per Output Token): {decode_time * 1000 / (output_tokens - 1):.3f} ms/token"
    )
    logger.success(f"E2E Latency (End-to-End Latency): {total_time:.3f} seconds")
    logger.success(
        f"E2E TPS (End-to-End Tokens Per Second): {output_tokens / total_time:.2f} tokens/s"
    )
