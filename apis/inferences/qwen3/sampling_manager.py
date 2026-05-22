#!/usr/bin/python3
# -*- coding: utf-8 -*-

# Copyright (c) 2025 HOUMO AI
#
# File: sampling_manager.py
# Description:
#   Sampling Manager for Qwen3 - Implements various sampling strategies including
#   top-k, top-p, temperature scaling, and repetition penalty for text generation.
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

import numpy as np
from typing import List, Optional


class SamplingManager:
    """
    Manages different sampling strategies for text generation.

    This class provides methods for applying various sampling techniques like
    temperature scaling, top-k, top-p (nucleus sampling), and repetition penalty
    to control the randomness and quality of generated text.
    """

    def __init__(
        self,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        min_tokens_to_keep: int = 1,
    ):
        """
        Initialize the sampling manager with specified parameters.

        Args:
            temperature: Controls randomness of predictions (higher = more random)
            top_k: Keep only k most likely tokens (None means no limit)
            top_p: Keep tokens with cumulative probability up to p (nucleus sampling)
            repetition_penalty: Penalty for repeated tokens (1.0 = no penalty)
            min_tokens_to_keep: Minimum number of tokens to keep during top-p filtering
        """
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.min_tokens_to_keep = min_tokens_to_keep

    def softmax(self, x: np.ndarray) -> np.ndarray:
        """
        Compute softmax values for array x.

        Args:
            x: Input array of logits

        Returns:
            Softmax-normalized probabilities
        """
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def apply_temperature(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to logits.

        Args:
            logits: Input logits array

        Returns:
            Temperature-adjusted logits
        """
        if self.temperature <= 0:
            raise ValueError("Temperature must larger than 0")

        return logits / self.temperature

    def apply_repetition_penalty(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Apply repetition penalty to logits based on previously generated tokens.

        Args:
            logits: Input logits array
            previous_tokens: List of previously generated token IDs

        Returns:
            Repetition penalty-adjusted logits
        """
        if self.repetition_penalty == 1.0 or not previous_tokens:
            return logits

        adjusted_logits = logits.copy()
        # Apply penalty to each unique token in the history
        for token_id in set(previous_tokens):
            if 0 <= token_id < len(logits):
                if logits[token_id] < 0:
                    adjusted_logits[token_id] = (
                        logits[token_id] * self.repetition_penalty
                    )
                else:
                    adjusted_logits[token_id] = (
                        logits[token_id] / self.repetition_penalty
                    )

        return adjusted_logits

    def apply_top_k(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply top-k sampling to probability distribution.

        Args:
            probs: Probability distribution

        Returns:
            Top-k filtered probability distribution
        """
        if self.top_k is None or self.top_k <= 0:
            return probs

        top_k = min(self.top_k, len(probs))
        if top_k <= 0:
            return probs

        # Find indices of top-k probabilities
        top_k_indices = np.argpartition(probs, -top_k)[-top_k:]
        # Create mask to zero out non-top-k probabilities
        mask = np.ones_like(probs, dtype=bool)
        mask[top_k_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        # Renormalize the probabilities
        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def apply_top_p(self, probs: np.ndarray) -> np.ndarray:
        """
        Apply top-p (nucleus) sampling to probability distribution.

        Args:
            probs: Probability distribution

        Returns:
            Top-p filtered probability distribution
        """
        if self.top_p >= 1.0:
            return probs

        # Sort probabilities in descending order
        sorted_indices = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_indices]

        cumulative_probs = np.cumsum(sorted_probs)
        cutoff_indices = np.where(cumulative_probs >= self.top_p)[0]

        if len(cutoff_indices) > 0:
            cutoff_index = cutoff_indices[0]
            if cutoff_index < self.min_tokens_to_keep - 1:
                cutoff_index = self.min_tokens_to_keep - 1

            selected_indices = sorted_indices[: cutoff_index + 1]
        else:
            selected_indices = sorted_indices

        # Zero out probabilities outside the selected set
        mask = np.ones_like(probs, dtype=bool)
        mask[selected_indices] = False
        filtered_probs = probs.copy()
        filtered_probs[mask] = 0

        # Renormalize the probabilities
        if np.sum(filtered_probs) > 0:
            normalized_probs = filtered_probs / np.sum(filtered_probs)
        else:
            normalized_probs = np.ones_like(probs) / len(probs)

        return normalized_probs

    def process_logits(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Process raw logits through all configured sampling techniques.

        Args:
            logits: Raw logits from the model
            previous_tokens: List of previously generated token IDs

        Returns:
            Processed probability distribution ready for sampling
        """
        processed_logits = logits.copy()
        # Apply repetition penalty
        processed_logits = self.apply_repetition_penalty(
            processed_logits, previous_tokens
        )

        # Apply softmax
        # Note: Using direct processing instead of softmax for performance
        probs = processed_logits
        # probs = self.softmax(processed_logits)

        # Apply top-k filtering
        probs = self.apply_top_k(probs)

        # Apply top-p filtering
        probs = self.apply_top_p(probs)

        # Apply temperature scaling
        probs = self.apply_temperature(probs)

        return probs

    def sample(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> int:
        """
        Sample a token from the processed logits.

        Args:
            logits: Raw logits from the model
            previous_tokens: List of previously generated token IDs

        Returns:
            Sampled token ID as a numpy array
        """
        logits = logits[0]
        probs = self.process_logits(logits, previous_tokens)
        # Handle edge case where all probabilities are zero
        if np.all(probs == 0):
            probs = np.ones_like(probs) / len(probs)

        # sampled_index = np.random.choice(len(probs), p=probs)
        sampled_index = probs.argmax(-1)

        return np.array([[sampled_index]])

    def get_processed_probs(
        self, logits: np.ndarray, previous_tokens: Optional[List[int]] = None
    ) -> np.ndarray:
        """
        Get the processed probability distribution without sampling.

        Args:
            logits: Raw logits from the model
            previous_tokens: List of previously generated token IDs

        Returns:
            Processed probability distribution
        """
        return self.process_logits(logits, previous_tokens)
