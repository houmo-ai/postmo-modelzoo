/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_types.h
 * Description:
 *   Shared Qwen3-TTS C++ tensor and hidden-sequence data types.
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

#include <cstddef>
#include <stdexcept>
#include <vector>

#include "base/houmo.h"

namespace houmo {

struct Qwen3TTSHiddenSequence {
  std::vector<float16> data;
  size_t sequence_length = 0;
  size_t hidden_dim = 0;

  void Validate() const {
    if (data.size() != sequence_length * hidden_dim) {
      throw std::invalid_argument("Hidden sequence data does not match its shape");
    }
  }
};

}  // namespace houmo
