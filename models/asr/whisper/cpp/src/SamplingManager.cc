/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: SamplingManager.cc
 * Description:
 *   Implementation of SamplingManager for text generation post-processing.
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

#include "SamplingManager.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <utility>
#include <vector>

void SamplingManager::softmax(const float* logits, size_t size,
                              float* probs) const {
  if (size == 0) return;

  // Find max for numerical stability
  float max_val = logits[0];
  for (size_t i = 1; i < size; ++i) {
    if (logits[i] > max_val) max_val = logits[i];
  }

  // Compute exp(x - max) and sum
  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    probs[i] = std::exp(logits[i] - max_val);
    sum += probs[i];
  }

  // Normalize
  if (sum > 0.0f) {
    for (size_t i = 0; i < size; ++i) {
      probs[i] /= sum;
    }
  }
}

void SamplingManager::applyTemperature(float* logits, size_t size) const {
  if (temperature_ <= 0.0f) {
    // Invalid temperature, skip
    return;
  }

  for (size_t i = 0; i < size; ++i) {
    logits[i] /= temperature_;
  }
}

void SamplingManager::applyRepetitionPenalty(
    float* logits, size_t size, const std::vector<int>& previous_tokens) const {
  if (repetition_penalty_ == 1.0f || previous_tokens.empty()) {
    return;
  }

  // Create unique set of previous tokens
  std::vector<int> unique_tokens = previous_tokens;
  std::sort(unique_tokens.begin(), unique_tokens.end());
  unique_tokens.erase(std::unique(unique_tokens.begin(), unique_tokens.end()),
                      unique_tokens.end());

  for (int token_id : unique_tokens) {
    if (token_id >= 0 && static_cast<size_t>(token_id) < size) {
      if (logits[token_id] < 0) {
        // Negative logits: multiply by penalty
        logits[token_id] *= repetition_penalty_;
      } else {
        // Positive logits: divide by penalty
        logits[token_id] /= repetition_penalty_;
      }
    }
  }
}

void SamplingManager::applyPresenceRepetitionPenalty(
    float* logits, size_t size, const std::vector<int>& previous_tokens) const {
  if (repetition_penalty_ == 0.0f || previous_tokens.empty()) {
    return;
  }

  // Create unique set of previous tokens
  std::vector<int> unique_tokens = previous_tokens;
  std::sort(unique_tokens.begin(), unique_tokens.end());
  unique_tokens.erase(std::unique(unique_tokens.begin(), unique_tokens.end()),
                      unique_tokens.end());

  // Apply presence penalty: subtract penalty from logits of repeated tokens
  for (int token_id : unique_tokens) {
    if (token_id >= 0 && static_cast<size_t>(token_id) < size) {
      logits[token_id] -= repetition_penalty_;
    }
  }
}

void SamplingManager::applyTopK(float* probs, size_t size) const {
  if (top_k_ <= 0 || static_cast<size_t>(top_k_) >= size) {
    return;  // Disabled or top_k >= vocab_size
  }

  size_t k = std::min(static_cast<size_t>(top_k_), size);

  // Find the k-th largest value using partial sort
  std::vector<float> temp_probs(probs, probs + size);
  std::nth_element(temp_probs.begin(), temp_probs.begin() + (size - k),
                   temp_probs.end(), std::greater<float>());
  float threshold = temp_probs[size - k];

  // Filter: set all values below threshold to 0
  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    if (probs[i] < threshold) {
      probs[i] = 0.0f;
    } else {
      sum += probs[i];
    }
  }

  // Renormalize
  if (sum > 0.0f) {
    for (size_t i = 0; i < size; ++i) {
      if (probs[i] > 0.0f) {
        probs[i] /= sum;
      }
    }
  } else {
    // Fallback: uniform distribution
    for (size_t i = 0; i < size; ++i) {
      probs[i] = 1.0f / size;
    }
  }
}

void SamplingManager::applyTopP(float* probs, size_t size) const {
  if (top_p_ >= 1.0f) {
    return;  // Disabled
  }

  // Create index-probability pairs for sorting
  std::vector<std::pair<float, size_t>> indexed_probs(size);
  for (size_t i = 0; i < size; ++i) {
    indexed_probs[i] = {probs[i], i};
  }

  // Sort by probability descending
  std::sort(
      indexed_probs.begin(), indexed_probs.end(),
      [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        return a.first > b.first;
      });

  // Find cutoff index where cumulative probability >= top_p
  float cumulative = 0.0f;
  size_t cutoff_index = 0;
  for (size_t i = 0; i < size; ++i) {
    cumulative += indexed_probs[i].first;
    cutoff_index = i;
    if (cumulative >= top_p_) {
      break;
    }
  }

  // Ensure minimum tokens to keep
  size_t min_keep = static_cast<size_t>(min_tokens_to_keep_);
  if (cutoff_index < min_keep - 1) {
    cutoff_index = min_keep - 1;
  }

  // Build mask for tokens to keep
  std::vector<bool> keep_mask(size, false);
  for (size_t i = 0; i <= cutoff_index; ++i) {
    keep_mask[indexed_probs[i].second] = true;
  }

  // Filter: set values outside top-p to 0
  float sum = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    if (!keep_mask[i]) {
      probs[i] = 0.0f;
    } else {
      sum += probs[i];
    }
  }

  // Renormalize
  if (sum > 0.0f) {
    for (size_t i = 0; i < size; ++i) {
      if (probs[i] > 0.0f) {
        probs[i] /= sum;
      }
    }
  } else {
    // Fallback: uniform distribution
    for (size_t i = 0; i < size; ++i) {
      probs[i] = 1.0f / size;
    }
  }
}

void SamplingManager::processLogits(const float* logits, size_t size,
                                    const std::vector<int>& previous_tokens,
                                    float* processed_probs) const {
  // Copy logits to working array
  std::vector<float> working_logits(logits, logits + size);

  // 1. Apply presence repetition penalty (matching Python implementation)
  applyPresenceRepetitionPenalty(working_logits.data(), size, previous_tokens);

  // 2. Use logits directly as probs (matching Python: not using softmax)
  // Python comment: "not using softmax in case of long time cost"
  std::copy(working_logits.begin(), working_logits.end(), processed_probs);

  // 3. Apply top-K
  applyTopK(processed_probs, size);

  // 4. Apply top-P
  applyTopP(processed_probs, size);

  // 5. Apply temperature
  applyTemperature(processed_probs, size);
}

int SamplingManager::sample(const float* logits, size_t size,
                            const std::vector<int>& previous_tokens) const {
  std::vector<float> probs(size);
  processLogits(logits, size, previous_tokens, probs.data());

  // Handle all-zero case
  bool all_zero = true;
  for (size_t i = 0; i < size; ++i) {
    if (probs[i] != 0.0f) {
      all_zero = false;
      break;
    }
  }
  if (all_zero) {
    for (size_t i = 0; i < size; ++i) {
      probs[i] = 1.0f / size;
    }
  }

  // Greedy sampling: return argmax (matching Python implementation)
  int sampled_index = 0;
  float max_prob = probs[0];
  for (size_t i = 1; i < size; ++i) {
    if (probs[i] > max_prob) {
      max_prob = probs[i];
      sampled_index = static_cast<int>(i);
    }
  }

  return sampled_index;
}

void SamplingManager::getProcessedProbs(const float* logits, size_t size,
                                        const std::vector<int>& previous_tokens,
                                        float* probs) const {
  processLogits(logits, size, previous_tokens, probs);
}