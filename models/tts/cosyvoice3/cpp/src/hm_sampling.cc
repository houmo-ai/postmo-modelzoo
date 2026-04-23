/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_sampling.cc
 * Description:
 *   Sampling strategies implementation for CosyVoice3 TTS LLM inference.
 *   Implements nucleus sampling and RAS (Repetition-Aware Sampling).
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

#include "hm_sampling.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#include "common_types.h"
#include "tcim_runtime_utils.h"

namespace houmo {

HmSampling::HmSampling(int top_k, float top_p, float tau_r, int win_size,
                       int speech_token_size)
    : top_k_(top_k),
      top_p_(top_p),
      tau_r_(tau_r),
      win_size_(win_size),
      speech_token_size_(speech_token_size),
      rng_(std::random_device{}()) {
  // Initialize default stop tokens (usually 6561 to 6760)
  for (int i = 0; i < 200; ++i) {
    stop_token_ids_.push_back(speech_token_size_ + i);
  }
}

HmSampling::~HmSampling() {}

void HmSampling::SetSeed(uint32_t seed) { rng_.seed(seed); }

void HmSampling::SetNextRandomValue(float random_value) {
  forced_random_value_ = std::clamp(random_value, 0.0f, 1.0f);
  has_forced_random_value_ = true;
}

void HmSampling::SetStopTokens(const std::vector<int>& stop_tokens) {
  stop_token_ids_ = stop_tokens;
}

float HmSampling::DrawRandom01() {
  if (has_forced_random_value_) {
    has_forced_random_value_ = false;
    return forced_random_value_;
  }
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  return dist(rng_);
}

int HmSampling::SampleFromWeights(const std::vector<float>& weights,
                                  const std::vector<int>& indices) {
  if (weights.empty() || indices.empty() || weights.size() != indices.size()) {
    throw std::runtime_error("Invalid weights for sampling.");
  }

  const float total_weight =
      std::accumulate(weights.begin(), weights.end(), 0.0f);
  if (total_weight <= 0.0f) {
    return indices.front();
  }

  const float threshold = DrawRandom01() * total_weight;
  float cumulative = 0.0f;
  for (size_t i = 0; i < weights.size(); ++i) {
    cumulative += weights[i];
    if (threshold < cumulative) {
      return indices[i];
    }
  }
  return indices.back();
}

std::vector<float> HmSampling::TensorToLogits(const tcim::Tensor& logits,
                                              int logits_dim) const {
  const auto* logits_data =
      static_cast<const TensorType*>(logits.Buffer().Data());
  const size_t num_elements = logits.MemSize() / sizeof(TensorType);
  if (logits_dim <= 0 || static_cast<size_t>(logits_dim) > num_elements) {
    throw std::runtime_error("Invalid logits_dim for sampling tensor.");
  }
  const size_t start_idx = num_elements - static_cast<size_t>(logits_dim);

  std::vector<float> converted_logits;
  converted_logits.reserve(static_cast<size_t>(logits_dim));
  for (size_t i = start_idx; i < num_elements; ++i) {
    converted_logits.push_back(static_cast<float>(logits_data[i]));
  }
  return converted_logits;
}

std::vector<float> HmSampling::Softmax(const std::vector<float>& logits) {
  if (logits.empty()) return {};

  float max_val = logits[0];
  for (size_t i = 1; i < logits.size(); ++i) {
    if (logits[i] > max_val) max_val = logits[i];
  }

  std::vector<float> probs(logits.size());
  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    probs[i] = std::exp(logits[i] - max_val);
    sum += probs[i];
  }

  if (sum > 0.0f) {
    for (size_t i = 0; i < probs.size(); ++i) {
      probs[i] /= sum;
    }
  }
  return probs;
}

int HmSampling::NucleusSampling(const std::vector<float>& logits) {
  auto probs = Softmax(logits);

  std::vector<int> indices(probs.size());
  std::iota(indices.begin(), indices.end(), 0);

  std::stable_sort(indices.begin(), indices.end(),
                   [&probs](int a, int b) { return probs[a] > probs[b]; });

  std::vector<int> truncated_indices;
  std::vector<float> truncated_probs;
  float cum_prob = 0.0f;

  for (size_t i = 0; i < indices.size() && cum_prob < top_p_ &&
                     static_cast<int>(truncated_indices.size()) < top_k_;
       ++i) {
    int idx = indices[i];
    cum_prob += probs[idx];
    truncated_indices.push_back(idx);
    truncated_probs.push_back(probs[idx]);
  }

  if (truncated_indices.empty()) {
    int max_idx = 0;
    float max_val = probs[0];
    for (size_t i = 1; i < probs.size(); ++i) {
      if (probs[i] > max_val) {
        max_val = probs[i];
        max_idx = static_cast<int>(i);
      }
    }
    return max_idx;
  }

  return SampleFromWeights(truncated_probs, truncated_indices);
}

int HmSampling::RandomSampling(const std::vector<float>& logits, int sampling) {
  (void)sampling;
  auto probs = Softmax(logits);
  std::vector<int> indices(probs.size());
  std::iota(indices.begin(), indices.end(), 0);
  return SampleFromWeights(probs, indices);
}

int HmSampling::RasSampling(const std::vector<float>& logits,
                            const std::vector<int>& decoded_tokens,
                            int sampling) {
  int top_ids = NucleusSampling(logits);

  int window_start =
      std::max(0, static_cast<int>(decoded_tokens.size()) - win_size_);
  int rep_num = 0;

  for (size_t i = window_start; i < decoded_tokens.size(); ++i) {
    if (decoded_tokens[i] == top_ids) {
      rep_num++;
    }
  }

  if (rep_num >= static_cast<int>(win_size_ * tau_r_)) {
    top_ids = RandomSampling(logits, sampling);
  }

  return top_ids;
}

int HmSampling::SamplingIds(const tcim::Tensor& logits,
                            const std::vector<int>& decoded_tokens,
                            int logits_dim, int sampling, bool ignore_eos) {
  const auto logits_vector = TensorToLogits(logits, logits_dim);
  int num_trials = 0;
  const int max_trials = 100;

  while (true) {
    int top_ids = RasSampling(logits_vector, decoded_tokens, sampling);

    if (!ignore_eos || top_ids != speech_token_size_) {
      return top_ids;
    }

    num_trials++;
    if (num_trials > max_trials) {
      std::cerr << "Warning: Sampling reached max_trials " << max_trials
                << " with EOS when ignore_eos=True\n";
      return top_ids;
    }
  }
}

}  // namespace houmo
