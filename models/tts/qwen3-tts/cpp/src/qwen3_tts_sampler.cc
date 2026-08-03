/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_sampler.cc
 * Description:
 *   Qwen3-TTS token sampling implementation.
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

#include "qwen3_tts_sampler.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

namespace houmo {

namespace {

void ApplyRepetitionPenalty(std::vector<float>* scores,
                            const std::vector<Token>& generated_tokens,
                            float repetition_penalty) {
  if (repetition_penalty == 1.0f) return;
  for (Token token : generated_tokens) {
    if (token < 0 || static_cast<size_t>(token) >= scores->size()) continue;
    float& score = (*scores)[static_cast<size_t>(token)];
    score = score < 0.0f ? score * repetition_penalty
                         : score / repetition_penalty;
  }
}

void SuppressTokens(std::vector<float>* scores,
                    const std::vector<Token>& suppress_tokens,
                    Token eos_token_id, size_t generated_count,
                    int min_new_tokens) {
  const float negative_infinity = -std::numeric_limits<float>::infinity();
  for (Token token : suppress_tokens) {
    if (token >= 0 && static_cast<size_t>(token) < scores->size()) {
      (*scores)[static_cast<size_t>(token)] = negative_infinity;
    }
  }
  if (eos_token_id >= 0 &&
      generated_count < static_cast<size_t>(min_new_tokens) &&
      static_cast<size_t>(eos_token_id) < scores->size()) {
    (*scores)[static_cast<size_t>(eos_token_id)] = negative_infinity;
  }
}

std::vector<size_t> SelectTopK(const std::vector<float>& scores, int top_k) {
  std::vector<size_t> indices(scores.size());
  std::iota(indices.begin(), indices.end(), 0);
  std::sort(indices.begin(), indices.end(), [&](size_t left, size_t right) {
    return scores[left] > scores[right];
  });
  if (top_k > 0 && static_cast<size_t>(top_k) < indices.size()) {
    indices.resize(static_cast<size_t>(top_k));
  }
  return indices;
}

void ApplyTopP(float top_p, std::vector<size_t>* indices,
               std::vector<double>* weights) {
  if (top_p >= 1.0f) return;
  const double total = std::accumulate(weights->begin(), weights->end(), 0.0);
  double cumulative = 0.0;
  size_t keep = 0;
  while (keep < weights->size()) {
    cumulative += (*weights)[keep];
    ++keep;
    if (cumulative / total >= top_p) break;
  }
  indices->resize(keep);
  weights->resize(keep);
}

std::vector<double> SoftmaxWeights(const std::vector<float>& scores,
                                   const std::vector<size_t>& indices) {
  const float max_score = scores[indices.front()];
  std::vector<double> weights;
  weights.reserve(indices.size());
  for (size_t index : indices) {
    weights.push_back(std::exp(static_cast<double>(scores[index] - max_score)));
  }
  return weights;
}

}  // namespace

Qwen3TTSSampler::Qwen3TTSSampler(
    Qwen3TTSSamplingConfig config, std::shared_ptr<std::mt19937> random)
    : config_(std::move(config)),
      random_(random != nullptr ? std::move(random)
                                : std::make_shared<std::mt19937>(config_.seed)) {
  if (config_.temperature <= 0.0f || config_.top_p <= 0.0f ||
      config_.top_p > 1.0f || config_.repetition_penalty <= 0.0f) {
    throw std::invalid_argument("Invalid sampling configuration");
  }
}

Token Qwen3TTSSampler::Sample(
    const std::vector<float16>& logits,
    const std::vector<Token>& generated_tokens) {
  if (logits.empty()) throw std::invalid_argument("Cannot sample empty logits");
  std::vector<float> scores(logits.size());
  std::transform(logits.begin(), logits.end(), scores.begin(),
                 [](float16 value) { return static_cast<float>(value); });
  ApplyRepetitionPenalty(&scores, generated_tokens,
                         config_.repetition_penalty);
  SuppressTokens(&scores, config_.suppress_tokens, config_.eos_token_id,
                 generated_tokens.size(), config_.min_new_tokens);
  for (float& score : scores) score /= config_.temperature;

  if (!config_.do_sample) {
    return static_cast<Token>(
        std::distance(scores.begin(),
                      std::max_element(scores.begin(), scores.end())));
  }

  std::vector<size_t> indices = SelectTopK(scores, config_.top_k);
  std::vector<double> weights = SoftmaxWeights(scores, indices);
  ApplyTopP(config_.top_p, &indices, &weights);
  std::discrete_distribution<size_t> distribution(weights.begin(),
                                                   weights.end());
  return static_cast<Token>(indices[distribution(*random_)]);
}

}  // namespace houmo
