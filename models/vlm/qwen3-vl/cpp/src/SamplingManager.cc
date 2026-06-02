/*
 * Copyright (c) 2026 HOUMO AI
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
#include <deque>
#include <functional>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <stdexcept>
#include <unordered_map>
#include <utility>
#include <vector>

// Hash function for N-gram
size_t SamplingManager::hashNgram(const std::vector<int>& ngram) const {
  size_t seed = 0;
  for (int token : ngram) {
    seed ^= std::hash<int>{}(token) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  }
  return seed;
}

void SamplingManager::resetNgramState() {
  ngram_pool_.clear();
  repeat_ngram_pool_.clear();
  repeat_triggered_ = false;
  consecutive_trigger_count_ = 0;
}

void SamplingManager::updateNgramPools(const std::vector<int>& history_tokens) {
  if (history_tokens.empty()) return;

  // Clear and rebuild pools from full history (no sliding window)
  ngram_pool_.clear();
  repeat_ngram_pool_.clear();

  // Build N-gram pools from all history tokens
  // ngram_pool_: for blocking (uses no_repeat_ngram_size_)
  if (no_repeat_ngram_size_ > 0 &&
      history_tokens.size() >= static_cast<size_t>(no_repeat_ngram_size_)) {
    for (size_t i = 0; i <= history_tokens.size() - no_repeat_ngram_size_;
         ++i) {
      std::vector<int> ngram(
          history_tokens.begin() + i,
          history_tokens.begin() + i + no_repeat_ngram_size_);
      size_t hash = hashNgram(ngram);
      ngram_pool_[hash]++;
    }
  }

  // repeat_ngram_pool_: for triggering (uses repeat_ngram_size_)
  if (repeat_ngram_size_ > 0 &&
      history_tokens.size() >= static_cast<size_t>(repeat_ngram_size_)) {
    for (size_t i = 0; i <= history_tokens.size() - repeat_ngram_size_; ++i) {
      std::vector<int> ngram(history_tokens.begin() + i,
                             history_tokens.begin() + i + repeat_ngram_size_);
      size_t hash = hashNgram(ngram);
      repeat_ngram_pool_[hash]++;
    }
  }

  // Check if ANY N-gram in the history exceeds threshold
  bool any_exceeds = false;
  for (const auto& pair : repeat_ngram_pool_) {
    if (pair.second >= repeat_count_threshold_) {
      any_exceeds = true;
      break;
    }
  }

  if (any_exceeds) {
    consecutive_trigger_count_++;
    repeat_triggered_ = true;
  } else {
    repeat_triggered_ = false;
  }
}

bool SamplingManager::wouldCreateRepeatedNgram(
    int candidate_token, const std::vector<int>& previous_tokens) const {
  if (no_repeat_ngram_size_ <= 0 || !repeat_triggered_) {
    return false;
  }

  // Need at least ngram_size - 1 previous tokens
  if (previous_tokens.size() < static_cast<size_t>(no_repeat_ngram_size_ - 1)) {
    return false;
  }

  // Construct candidate N-gram
  std::vector<int> candidate_ngram;
  size_t start_idx = previous_tokens.size() - (no_repeat_ngram_size_ - 1);
  for (size_t i = start_idx; i < previous_tokens.size(); ++i) {
    candidate_ngram.push_back(previous_tokens[i]);
  }
  candidate_ngram.push_back(candidate_token);

  // Check if this N-gram exists in pool
  size_t hash = hashNgram(candidate_ngram);
  auto it = ngram_pool_.find(hash);
  return it != ngram_pool_.end() && it->second > 0;
}

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
    throw std::invalid_argument("Temperature must larger than 0");
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
  if (presence_penalty_ == 0.0f || previous_tokens.empty()) {
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
      logits[token_id] -= presence_penalty_;
    }
  }
}

void SamplingManager::applyTopK(float* probs, size_t size) const {
  if (top_k_ <= 0 || static_cast<size_t>(top_k_) >= size) {
    return;  // Disabled or top_k >= vocab_size
  }

  size_t k = std::min(static_cast<size_t>(top_k_), size);

  // Create index-probability pairs for sorting (matching Python argpartition
  // behavior)
  std::vector<std::pair<float, size_t>> indexed_probs(size);
  for (size_t i = 0; i < size; ++i) {
    indexed_probs[i] = {probs[i], i};
  }

  // Use nth_element to find top-k indices (equivalent to numpy.argpartition)
  std::nth_element(
      indexed_probs.begin(), indexed_probs.begin() + k, indexed_probs.end(),
      [](const std::pair<float, size_t>& a, const std::pair<float, size_t>& b) {
        return a.first > b.first;  // Descending order
      });

  // Build mask for top-k indices
  std::vector<bool> keep_mask(size, false);
  for (size_t i = 0; i < k; ++i) {
    keep_mask[indexed_probs[i].second] = true;
  }

  // Filter: set values outside top-k to 0
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
      probs[i] /= sum;
    }
  } else {
    // Fallback: uniform distribution
    for (size_t i = 0; i < size; ++i) {
      probs[i] = 1.0f / size;
    }
  }
}

void SamplingManager::applyTopP(float* probs, size_t size) const {
  if (top_p_ >= 1.0f || size == 0) {
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
  size_t min_keep =
      std::min(static_cast<size_t>(std::max(min_tokens_to_keep_, 1)), size);
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
      probs[i] /= sum;
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

  // 1. Apply repetition penalty
  applyRepetitionPenalty(working_logits.data(), size, previous_tokens);

  // 2. Apply presence penalty
  applyPresenceRepetitionPenalty(working_logits.data(), size, previous_tokens);

  // 3. Apply temperature on logits
  if (temperature_ != 1.0f) {
    applyTemperature(working_logits.data(), size);
  }

  // 4. Convert logits to probabilities using softmax
  softmax(working_logits.data(), size, processed_probs);

  // 5. Apply top-K
  applyTopK(processed_probs, size);

  // 6. Apply top-P
  applyTopP(processed_probs, size);
}

int SamplingManager::sample(const float* logits, size_t size,
                            const std::vector<int>& previous_tokens) const {
  // Greedy sampling fast-path
  if (top_k_ == 1 && !(no_repeat_ngram_size_ > 0 && repeat_triggered_)) {
    std::vector<float> working_logits(logits, logits + size);
    applyRepetitionPenalty(working_logits.data(), size, previous_tokens);
    applyPresenceRepetitionPenalty(working_logits.data(), size,
                                   previous_tokens);
    int sampled_index = 0;
    float max_logit = working_logits[0];
    for (size_t i = 1; i < size; ++i) {
      if (working_logits[i] > max_logit) {
        max_logit = working_logits[i];
        sampled_index = static_cast<int>(i);
      }
    }
    return sampled_index;
  }

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

  // Normalize sum to 1 to avoid numerical errors
  float sum_probs = 0.0f;
  for (size_t i = 0; i < size; ++i) {
    sum_probs += probs[i];
  }
  if (sum_probs > 0.0f) {
    for (size_t i = 0; i < size; ++i) {
      probs[i] /= sum_probs;
    }
  } else {
    for (size_t i = 0; i < size; ++i) {
      probs[i] = 1.0f / size;
    }
  }

  // N-gram repetition blocking (if enabled and triggered)
  // If N-gram blocking is enabled and triggered, skip candidates that would
  // create repeated N-grams
  if (no_repeat_ngram_size_ > 0 && repeat_triggered_) {
    // Create index-probability pairs for sorting
    std::vector<std::pair<float, size_t>> indexed_probs(size);
    for (size_t i = 0; i < size; ++i) {
      indexed_probs[i] = {probs[i], i};
    }

    // Sort by probability descending
    std::sort(
        indexed_probs.begin(), indexed_probs.end(),
        [](const std::pair<float, size_t>& a,
           const std::pair<float, size_t>& b) { return a.first > b.first; });

    // Try each candidate in order of probability
    for (const auto& [prob, idx] : indexed_probs) {
      int candidate_token = static_cast<int>(idx);
      // Skip candidates that would create a repeated N-gram
      if (!wouldCreateRepeatedNgram(candidate_token, previous_tokens)) {
        return candidate_token;
      }
    }

    // All candidates blocked, fallback to random sampling
  }

  // Random sampling according to probability distribution
  // Use thread-local random generator for efficiency
  static thread_local std::mt19937 gen(std::random_device{}());
  std::discrete_distribution<size_t> dist(probs.begin(), probs.end());
  return static_cast<int>(dist(gen));
}

void SamplingManager::getProcessedProbs(const float* logits, size_t size,
                                        const std::vector<int>& previous_tokens,
                                        float* probs) const {
  processLogits(logits, size, previous_tokens, probs);
}