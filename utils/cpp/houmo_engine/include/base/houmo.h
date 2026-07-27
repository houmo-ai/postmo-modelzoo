/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: houmo.h
 * Description:
 *   Houmo Inference Framework - Core type definitions including Token,
 *   ModelConfig, SamplingParams, and performance statistics structures.
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
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "half/half.hpp"

using float16 = half_float::half;

// ============================================================================
// Profiling toggle
// ============================================================================

// Enable profiling by default; disable via compile option if needed
#ifndef HOUOMO_ENABLE_PROFILING
#define HOUOMO_ENABLE_PROFILING 1
#endif

namespace houmo {

// ============================================================================
// Token type
// ============================================================================

using Token = int32_t;

constexpr Token TokenNull = -1;
constexpr Token TokenBos = -2;
constexpr Token TokenEos = -3;

// ============================================================================
// Exception
// ============================================================================

class Exception : public std::runtime_error {
 public:
  explicit Exception(const std::string& msg) : std::runtime_error(msg) {}
};

// ============================================================================
// Model type
// ============================================================================

enum class ModelType {
  LLM,
  VLM,
  ASR,
  TTS,
};

/**
 * @brief Model kind (used by Check function)
 */
enum class ModelKind {
  LLM,
  VLM,
  ASR,
  TTS,
};

// ============================================================================
// Check result
// ============================================================================

/**
 * @brief Configuration check result
 */
struct CheckResult {
  bool valid = false;
  std::string error_message;
};

// ============================================================================
// Model configuration
// ============================================================================

/**
 * @brief Model configuration structure
 *
 * Contains all paths and runtime parameters required for model loading.
 */
struct ModelConfig {
  // Runtime parameters
  std::vector<int> devices = {0};
  int batch_size = 1;
  bool lazy_mode = false;

  // Model paths
  std::string prefill_path;    ///< Prefill model path (.hmm)
  std::string decode_path;     ///< Decode model path (.hmm)
  std::string embedding_path;  ///< Embedding weight path (.bin)
  std::string tokenizer_path;  ///< Tokenizer vocabulary path (.json), optional
  std::string vision_path;     ///< Vision model path (.hmm), VLM only

  // Extension parameters
  std::map<std::string, std::string>
      extra_params;  ///< Extra params for subclasses
};

// ============================================================================
// Configuration check function
// ============================================================================

// ============================================================================
// Model metadata
// ============================================================================

struct ModelInfo {
  ModelType type;
  std::string model_name;  // e.g. "qwen3-4b", "qwen3.5-2b", "qwen3-vl-4b"
  int n_batch = 0;
  int n_vocab = 0;
  int n_embd = 0;
  int n_layer = 0;
  int n_ctx = 0;  // Context length
  int prefill_length = 0;
  int kv_cache_layers = 0;
  int n_logits = 0;
};

// ============================================================================
// Performance statistics
// ============================================================================

struct PerfStats {
  double prefill_time_ms = 0;
  double decode_time_ms = 0;
  double total_time_ms = 0;
  double ttft_ms = 0;            // Time to First Token
  double tpot_ms = 0;            // Time Per Output Token
  double tps = 0;                // Tokens Per Second
  double embedding_time_ms = 0;  // Vision encoding time

  int n_input_tokens = 0;
  int n_output_tokens = 0;

  size_t cpu_memory_used = 0;
  size_t npu_memory_used = 0;
  size_t kv_cache_size = 0;
};

// ============================================================================
// Sampling parameters
// ============================================================================

struct SamplingParams {
  float temperature = 1.0f;
  float top_p = 1.0f;
  int top_k = 1;
  float repetition_penalty = 1.0f;
  int penalty_last_n = 64;
  int max_tokens = 0;
  std::vector<Token> stop_tokens;
  float frequency_penalty = 0.0f;
  float presence_penalty = 1.5f;
  float min_p = 0.0f;
  bool greedy = false;

  // Tokenize options
  bool add_bos = false;
  bool add_eos = false;

  // asr specific
  std::string language = "auto";  // "auto" for detection, or specific language
                                  // code (e.g., "zh", "en")
};

// ============================================================================
// Version info
// ============================================================================

std::string version();
std::string build_info();

template <typename T>
static int eigen_argmax(const T* ptr, std::size_t n) {
  using Eigen::Tensor;
  using Eigen::TensorMap;

  TensorMap<Tensor<const T, 1>> tm(static_cast<const T*>(ptr), n);

  Eigen::Tensor<Eigen::Index, 0> t = tm.argmax();
  Eigen::Index idx = t(0);

  return static_cast<int>(idx);
}

}  // namespace houmo
