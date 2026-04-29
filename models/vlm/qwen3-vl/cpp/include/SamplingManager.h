/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: SamplingManager.h
 * Description:
 *   Sampling manager for text generation post-processing.
 *   Implements temperature, top-k, top-p, and repetition penalty sampling.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __SAMPLING_MANAGER_H__
#define __SAMPLING_MANAGER_H__

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <vector>

/**
 * @brief Sampling manager for post-processing logits in text generation
 *
 * This class implements various sampling strategies including:
 * - Temperature scaling
 * - Top-K filtering
 * - Top-P (nucleus) filtering
 * - Presence repetition penalty
 */
class SamplingManager {
 public:
  /**
   * @brief Construct a new Sampling Manager object
   *
   * @param temperature Temperature for scaling logits (>0)
   * @param top_k Number of top tokens to keep (-1 or 0 means disabled)
   * @param top_p Cumulative probability threshold for nucleus sampling (0-1)
   * @param repetition_penalty Penalty for repeated tokens (0 means disabled,
   * uses presence penalty)
   * @param min_tokens_to_keep Minimum number of tokens to keep in top-p
   */
  SamplingManager(float temperature = 1.0f, int top_k = -1, float top_p = 1.0f,
                  float repetition_penalty = 1.0f, int min_tokens_to_keep = 1)
      : temperature_(temperature),
        top_k_(top_k),
        top_p_(top_p),
        repetition_penalty_(repetition_penalty),
        min_tokens_to_keep_(min_tokens_to_keep) {}

  /**
   * @brief Apply softmax to convert logits to probabilities
   *
   * @param logits Input logits array
   * @param size Size of the array
   * @param probs Output probabilities array (must be pre-allocated)
   */
  void softmax(const float* logits, size_t size, float* probs) const;

  /**
   * @brief Apply temperature scaling to logits
   *
   * @param logits Input/output logits array
   * @param size Size of the array
   */
  void applyTemperature(float* logits, size_t size) const;

  /**
   * @brief Apply multiplicative repetition penalty to logits
   *
   * @param logits Input/output logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   */
  void applyRepetitionPenalty(float* logits, size_t size,
                              const std::vector<int>& previous_tokens) const;

  /**
   * @brief Apply presence repetition penalty (subtractive penalty)
   *
   * @param logits Input/output logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   */
  void applyPresenceRepetitionPenalty(
      float* logits, size_t size,
      const std::vector<int>& previous_tokens) const;

  /**
   * @brief Apply top-K filtering to probabilities
   *
   * @param probs Input/output probabilities array
   * @param size Size of the array
   */
  void applyTopK(float* probs, size_t size) const;

  /**
   * @brief Apply top-P (nucleus) filtering to probabilities
   *
   * @param probs Input/output probabilities array
   * @param size Size of the array
   */
  void applyTopP(float* probs, size_t size) const;

  /**
   * @brief Process logits through all sampling steps
   *
   * Order: repetition_penalty -> top_k -> top_p -> temperature
   *
   * @param logits Input logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   * @param processed_probs Output processed probabilities (must be
   * pre-allocated)
   */
  void processLogits(const float* logits, size_t size,
                     const std::vector<int>& previous_tokens,
                     float* processed_probs) const;

  /**
   * @brief Sample a token from processed logits
   *
   * @param logits Input logits array (shape: [1][vocab_size])
   * @param size Size of the logits array (vocab_size)
   * @param previous_tokens List of previously generated token IDs
   * @return Sampled token ID
   */
  int sample(const float* logits, size_t size,
             const std::vector<int>& previous_tokens) const;

  /**
   * @brief Get processed probabilities without sampling
   *
   * @param logits Input logits array
   * @param size Size of the array
   * @param previous_tokens List of previously generated token IDs
   * @param probs Output probabilities (must be pre-allocated)
   */
  void getProcessedProbs(const float* logits, size_t size,
                         const std::vector<int>& previous_tokens,
                         float* probs) const;

  // Getters
  float getTemperature() const { return temperature_; }
  int getTopK() const { return top_k_; }
  float getTopP() const { return top_p_; }
  float getRepetitionPenalty() const { return repetition_penalty_; }
  int getMinTokensToKeep() const { return min_tokens_to_keep_; }

  // Setters
  void setTemperature(float temp) { temperature_ = temp; }
  void setTopK(int k) { top_k_ = k; }
  void setTopP(float p) { top_p_ = p; }
  void setRepetitionPenalty(float penalty) { repetition_penalty_ = penalty; }
  void setMinTokensToKeep(int min_keep) { min_tokens_to_keep_ = min_keep; }

 private:
  float temperature_;         // Temperature for scaling (>0)
  int top_k_;                 // Top-K threshold (-1 or 0 means disabled)
  float top_p_;               // Top-P threshold (0-1, 1.0 means disabled)
  float repetition_penalty_;  // Repetition penalty (0 means use presence
                              // penalty)
  int min_tokens_to_keep_;    // Minimum tokens to keep in top-p
};

#endif  // __SAMPLING_MANAGER_H__