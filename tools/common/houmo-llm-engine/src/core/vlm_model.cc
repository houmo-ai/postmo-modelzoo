/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: vlm_model.cc
 * Description:
 *   VLM (Vision-Language Model) base class implementation.
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

#include "core/vlm_model.h"
#include "base/houmo.h"

namespace houmo {

// ============================================================================
// VLMModel implementation
// ============================================================================

VLMModel::VLMModel(const ModelConfig& config) : LLMModel(config) {
  // TODO: Initialize vision encoder parameters
}

std::vector<float16> VLMModel::encode_image(const uint8_t* image_data,
                                            int width,
                                            int height,
                                            int channels) {
  // TODO: Implement image encoding
  // 1. Image preprocessing (resize, normalize)
  // 2. Run vision encoder
  // 3. Return vision embeddings
  throw Exception("VLMModel::encode_image() not implemented in base class");
}

std::unique_ptr<Context> VLMModel::create_context(int n_ctx) {
  // Base class does not support creating Context; subclasses must override this method
  throw Exception("VLMModel::create_context() must be overridden by subclass");
}

}  // namespace houmo
