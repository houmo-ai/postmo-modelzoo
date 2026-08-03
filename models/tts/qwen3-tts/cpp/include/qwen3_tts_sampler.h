/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_sampler.h
 * Description:
 *   Qwen3-TTS token sampling configuration and interface.
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

#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include "base/houmo.h"

namespace houmo {

struct Qwen3TTSSamplingConfig {
  bool do_sample = true;
  float temperature = 0.9f;
  int top_k = 50;
  float top_p = 1.0f;
  float repetition_penalty = 1.0f;
  int min_new_tokens = 0;
  Token eos_token_id = -1;
  std::vector<Token> suppress_tokens;
  uint32_t seed = 0;
};

class Qwen3TTSSampler {
 public:
  explicit Qwen3TTSSampler(
      Qwen3TTSSamplingConfig config,
      std::shared_ptr<std::mt19937> random = nullptr);

  Token Sample(const std::vector<float16>& logits,
               const std::vector<Token>& generated_tokens);

 private:
  Qwen3TTSSamplingConfig config_;
  std::shared_ptr<std::mt19937> random_;
};

}  // namespace houmo
