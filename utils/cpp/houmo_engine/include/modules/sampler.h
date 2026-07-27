/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: sampler.h
 * Description:
 *   Token sampler implementing multiple sampling strategies including
 *   greedy, temperature, top-k, top-p, and repetition penalty.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <vector>

#include "base/houmo.h"

namespace houmo {

/**
 * @brief Token sampler
 *
 * Implements multiple sampling strategies: greedy, temperature, top-k,
 * top-p, repetition penalty, etc.
 *
 * Sampling pipeline (aligned with demo.py SamplingManager):
 * logits -> penalties -> top_k (before softmax) -> temperature -> softmax
 * -> top_p -> sample
 */
class Sampler {
 public:
  /**
   * @brief Constructor
   * @param params Sampling parameters
   */
  explicit Sampler(const SamplingParams& params);

  ~Sampler();

  // Non-copyable
  Sampler(const Sampler&) = delete;
  Sampler& operator=(const Sampler&) = delete;

  // Movable
  Sampler(Sampler&&) noexcept = default;
  Sampler& operator=(Sampler&&) noexcept = default;

  /**
   * @brief Sample a token from logits
   * @param logits Logits array
   * @param size Logits size
   * @return Sampled token ID
   */
  Token sample(const float16* logits, size_t size);

  /**
   * @brief Sample a token from logits (with repetition penalty)
   * @param logits Logits array
   * @param size Logits size
   * @param previous_tokens Previous token sequence (for repetition penalty)
   * @return Sampled token ID
   */
  Token sample(const float16* logits, size_t size,
               const std::vector<Token>& previous_tokens);

  /**
   * @brief Get processed probability distribution (for debugging/test alignment)
   * @param logits Logits array
   * @param size Logits size
   * @param previous_tokens Previous token sequence
   * @return Processed probability distribution
   */
  std::vector<float16> get_processed_probs(
      const float16* logits, size_t size,
      const std::vector<Token>& previous_tokens);

  /**
   * @brief Update sampling parameters
   * @param params New sampling parameters
   */
  void set_params(const SamplingParams& params);

  /**
   * @brief Get current sampling parameters
   */
  const SamplingParams& params() const { return params_; }

 private:
  // Apply combined penalties (repetition + presence) in a single pass
  void apply_penalties(float16* logits, size_t size,
                       const std::vector<Token>& previous_tokens);

  // Apply top-k on logits (before softmax, for performance)
  void apply_top_k_on_logits(float16* logits, size_t size);

  // Softmax
  void softmax(float16* data, size_t size);

  // Apply top-p on probabilities
  void apply_top_p(float16* probs, size_t size);

  // Apply temperature on logits
  void apply_temperature(float16* logits, size_t size);

  // Helper function
  int argmax(const float16* data, size_t size) const;

  SamplingParams params_;
  int min_tokens_to_keep_ = 1;
};

}  // namespace houmo
