/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: sampler.cc
 * Description:
 *   Token sampler implementation with temperature, top-k, top-p, and penalties.
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

#include "modules/sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <unordered_set>

namespace houmo {

Sampler::Sampler(const SamplingParams& params)
    : params_(params), min_tokens_to_keep_(1) {}

Sampler::~Sampler() = default;

Token Sampler::sample(const float16* logits, size_t size) {
  std::vector<Token> empty_tokens;
  return sample(logits, size, empty_tokens);
}

Token Sampler::sample(const float16* logits, size_t size,
                      const std::vector<Token>& previous_tokens) {
  // Greedy fast-path: top_k == 1, no softmax needed
  if (params_.top_k == 1) {
    std::vector<float16> processed(logits, logits + size);
    apply_penalties(processed.data(), size, previous_tokens);
    return argmax(processed.data(), size);
  }

  // Normal path with full processing
  auto probs = get_processed_probs(logits, size, previous_tokens);
  return argmax(probs.data(), size);
}

std::vector<float16> Sampler::get_processed_probs(
    const float16* logits, size_t size,
    const std::vector<Token>& previous_tokens) {
  // Copy logits
  std::vector<float16> processed(logits, logits + size);

  // 1. Apply penalties (repetition + presence merged into single pass)
  apply_penalties(processed.data(), size, previous_tokens);

  // 2. Apply top-k on logits (before softmax, optimized for performance)
  apply_top_k_on_logits(processed.data(), size);

  // 3. Apply temperature on logits
  apply_temperature(processed.data(), size);

  // 4. Apply softmax to convert to probabilities
  softmax(processed.data(), size);

  // 5. Apply top-p on probabilities
  apply_top_p(processed.data(), size);

  return processed;
}

void Sampler::set_params(const SamplingParams& params) { params_ = params; }

void Sampler::apply_penalties(float16* logits, size_t size,
                              const std::vector<Token>& previous_tokens) {
  if (previous_tokens.empty()) {
    return;
  }

  bool need_repetition = (params_.repetition_penalty != 1.0f);
  bool need_presence = (params_.presence_penalty != 0.0f);

  if (!need_repetition && !need_presence) {
    return;
  }

  // Get unique tokens (single pass)
  std::unordered_set<Token> unique_tokens(previous_tokens.begin(),
                                          previous_tokens.end());

  // Single pass to apply both penalties
  for (Token token_id : unique_tokens) {
    if (token_id >= 0 && static_cast<size_t>(token_id) < size) {
      float16 logit = logits[token_id];

      // Apply repetition penalty
      if (need_repetition) {
        if (logit < 0) {
          // Negative: multiply by penalty
          logits[token_id] = logit * params_.repetition_penalty;
        } else {
          // Positive: divide by penalty
          logits[token_id] = logit / params_.repetition_penalty;
        }
        logit = logits[token_id];  // Update for presence penalty
      }

      // Apply presence penalty
      if (need_presence) {
        logits[token_id] = logit - params_.presence_penalty;
      }
    }
  }
}

void Sampler::apply_top_k_on_logits(float16* logits, size_t size) {
  if (params_.top_k <= 0 || static_cast<size_t>(params_.top_k) >= size) {
    return;
  }

  int top_k = std::min(params_.top_k, static_cast<int>(size));

  // Use nth_element to find the top_k largest indices
  std::vector<int> indices(size);
  std::iota(indices.begin(), indices.end(), 0);
  std::nth_element(indices.begin(), indices.begin() + top_k, indices.end(),
                   [&](int a, int b) { return logits[a] > logits[b]; });

  // Set the rest to -inf (becomes 0 after softmax)
  std::vector<bool> keep(size, false);
  for (int i = 0; i < top_k; i++) {
    keep[indices[i]] = true;
  }

  for (size_t i = 0; i < size; i++) {
    if (!keep[i]) {
      logits[i] = static_cast<float16>(-std::numeric_limits<float>::infinity());
    }
  }
}

void Sampler::softmax(float16* data, size_t size) {
  // Find max value (numerical stability)
  float16 max_val = data[0];
  for (size_t i = 1; i < size; i++) {
    if (data[i] > max_val) {
      max_val = data[i];
    }
  }

  // exp(x - max)
  float16 sum = static_cast<float16>(0.0f);
  for (size_t i = 0; i < size; i++) {
    data[i] =
        static_cast<float16>(std::exp(static_cast<float>(data[i] - max_val)));
    sum += data[i];
  }

  // Normalize
  if (sum > 0) {
    for (size_t i = 0; i < size; i++) {
      data[i] /= sum;
    }
  }
}

void Sampler::apply_top_p(float16* probs, size_t size) {
  if (params_.top_p >= 1.0f) {
    return;
  }

  // Sort indices by probability in descending order
  std::vector<int> sorted_indices(size);
  std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
  std::sort(sorted_indices.begin(), sorted_indices.end(),
            [&](int a, int b) { return probs[a] > probs[b]; });

  // Compute cumulative probability
  float16 cumulative = static_cast<float16>(0.0f);
  size_t cutoff_index = 0;

  for (size_t i = 0; i < size; i++) {
    cumulative += probs[sorted_indices[i]];
    if (cumulative >= params_.top_p) {
      cutoff_index = i;
      break;
    }
  }

  // Ensure minimum number of tokens are kept
  if (cutoff_index < static_cast<size_t>(min_tokens_to_keep_ - 1)) {
    cutoff_index = min_tokens_to_keep_ - 1;
  }

  // Create keep set
  std::vector<bool> keep(size, false);
  for (size_t i = 0; i <= cutoff_index; i++) {
    keep[sorted_indices[i]] = true;
  }

  // Zero out the rest
  float16 sum = static_cast<float16>(0.0f);
  for (size_t i = 0; i < size; i++) {
    if (!keep[i]) {
      probs[i] = static_cast<float16>(0.0f);
    }
    sum += probs[i];
  }

  // Normalize
  if (sum > 0) {
    for (size_t i = 0; i < size; i++) {
      probs[i] /= sum;
    }
  }
}

void Sampler::apply_temperature(float16* logits, size_t size) {
  if (params_.temperature <= 0.0f) {
    return;  // Avoid division by zero
  }

  if (params_.temperature == 1.0f) {
    return;  // No processing needed
  }

  for (size_t i = 0; i < size; i++) {
    logits[i] /= params_.temperature;
  }
}

int Sampler::argmax(const float16* data, size_t size) const {
  return eigen_argmax<float16>(data, size);
}

}  // namespace houmo
