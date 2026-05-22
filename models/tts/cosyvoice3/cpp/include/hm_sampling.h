/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_sampling.h
 * Description:
 *   Sampling strategies for CosyVoice3 TTS LLM inference.
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

#pragma once

#include <random>
#include <vector>

namespace tcim {
class Tensor;
}

namespace houmo {

class HmSampling {
 public:
  HmSampling(int top_k = 25, float top_p = 0.8f, float tau_r = 0.1f,
             int win_size = 10, int speech_token_size = 6561);
  ~HmSampling();

  void SetSeed(uint32_t seed);
  void SetNextRandomValue(float random_value);
  void SetStopTokens(const std::vector<int>& stop_tokens);

  int SamplingIds(const tcim::Tensor& logits,
                  const std::vector<int>& decoded_tokens, int logits_dim,
                  int sampling, bool ignore_eos);

 private:
  std::vector<float> TensorToLogits(const tcim::Tensor& logits,
                                    int logits_dim) const;

  int RasSampling(const std::vector<float>& logits,
                  const std::vector<int>& decoded_tokens, int sampling);

  int NucleusSampling(const std::vector<float>& logits);

  int RandomSampling(const std::vector<float>& logits, int sampling);

  int SampleFromWeights(const std::vector<float>& weights,
                        const std::vector<int>& indices);

  float DrawRandom01();

  std::vector<float> Softmax(const std::vector<float>& logits);

  int top_k_;
  float top_p_;
  float tau_r_;
  int win_size_;
  int speech_token_size_;
  std::vector<int> stop_token_ids_;
  std::mt19937 rng_;
  bool has_forced_random_value_ = false;
  float forced_random_value_ = 0.0f;
};

}  // namespace houmo
