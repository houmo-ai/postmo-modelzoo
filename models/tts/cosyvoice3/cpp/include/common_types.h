/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: common_types.h
 * Description:
 *   Common type definitions for CosyVoice3 TTS demo.
 *   Provides float16 tensor type and utility functions.
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

#ifndef COMMON_TYPES_H_
#define COMMON_TYPES_H_

#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "half.hpp"

namespace houmo {

// ============================================================================
// Type Definitions
// ============================================================================

/**
 * @brief Tensor element type used throughout CosyVoice3.
 *        All TCIM models use float16 (half precision).
 */
using TensorType = half_float::half;

/**
 * @brief Integer tensor type for indices, positions, etc.
 */
using IntTensorType = int32_t;

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * @brief Load file contents as bytes.
 * @param path File path to load.
 * @return File contents as string.
 * @throws std::runtime_error if file cannot be opened.
 */
inline std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << "\n";
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

/**
 * @brief Check TCIM status and throw on error.
 * @param status TCIM status code.
 * @param file Source file name.
 * @param line Source line number.
 * @throws std::runtime_error if status is not OK.
 */
inline void CheckTcimStatus(const int& status, const char* file = __FILE__,
                            int line = __LINE__) {
  if (status != 0) {
    std::ostringstream err_msg;
    err_msg << "TCIM error at " << file << ":" << line
            << ", status: " << status;
    throw std::runtime_error(err_msg.str());
  }
}

#define CHECK_TCIM_STATUS(status) CheckTcimStatus(status, __FILE__, __LINE__)

/**
 * @brief Convert float to TensorType (float16).
 */
inline TensorType ToTensorType(float value) {
  return static_cast<TensorType>(value);
}

/**
 * @brief Convert TensorType (float16) to float.
 */
inline float ToFloat(const TensorType& value) {
  return static_cast<float>(value);
}

/**
 * @brief Create a vector of TensorType filled with a value.
 */
inline std::vector<TensorType> CreateTensorVector(size_t size, float value) {
  return std::vector<TensorType>(size, static_cast<TensorType>(value));
}

/**
 * @brief Create a vector of IntTensorType filled with a value.
 */
inline std::vector<IntTensorType> CreateIntVector(size_t size, int value) {
  return std::vector<IntTensorType>(size, value);
}

// ============================================================================
// Tensor Info Structure
// ============================================================================

/**
 * @brief Tensor shape and type information.
 */
struct TensorInfo {
  std::vector<int64_t> shape;  ///< Tensor dimensions
  std::string dtype;           ///< Data type string
  size_t element_count;        ///< Total number of elements
  size_t byte_size;            ///< Total size in bytes

  /**
   * @brief Get shape as string for debugging.
   */
  std::string ShapeString() const {
    std::ostringstream ss;
    ss << "[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) ss << ", ";
      ss << shape[i];
    }
    ss << "]";
    return ss.str();
  }
};

// ============================================================================
// CosyVoice3FrontendInput - Model Input Structure
// ============================================================================

/**
 * @brief Input structure for TTS model inference.
 *
 * Contains all the inputs extracted from text and prompt audio:
 * - Text tokens for synthesis
 * - Prompt text tokens (for zero-shot)
 * - Prompt speech tokens (extracted from prompt audio)
 * - Prompt mel features (for Flow conditioning)
 * - Speaker embedding (for voice cloning)
 */
struct CosyVoice3FrontendInput {
  // prompt_text
  std::vector<int> prompt_text_tokens;
  int prompt_text_len;

  // llm_prompt_speech_token
  std::vector<int> llm_prompt_speech_tokens;
  int llm_prompt_speech_token_len;

  // flow_prompt_speech_token
  std::vector<int> flow_prompt_speech_tokens;
  int flow_prompt_speech_token_len;

  // prompt_speech_feat
  std::vector<TensorType> prompt_speech_feat;
  int prompt_speech_feat_len;

  // llm_embedding
  std::vector<TensorType> llm_embedding;

  // flow_embedding
  std::vector<TensorType> flow_embedding;

  // text
  std::vector<int> text_tokens;
  int text_len;
};

// ============================================================================
// CosyVoice3 Configuration
// ============================================================================

/**
 * @brief Configuration for CosyVoice3 models.
 *
 * Contains paths to all model files and runtime settings.
 * Reference: demo.py lines 1858-1903
 */
struct CosyVoice3Config {
  // Model directory (all .hmm files)
  std::string model_dir;

  // Embedding directory (all .hemm files)
  std::string embedding_dir;

  // Tokenizer directory
  std::string tokenizer_dir;

  // Output directory for generated audio
  std::string output_dir;

  // Individual model paths (optional, defaults to model_dir/*.hmm)
  std::string campplus_path;
  std::string speech_tokenizer_path;
  std::string llm_prefill_path;
  std::string llm_decode_path;
  std::string flow_spk_path;
  std::string flow_encoder_path;
  std::string flow_decoder_path;
  std::string hift_part1_path;
  std::string hift_part2_path;

  // Embedding paths (optional, defaults to embedding_dir/*.hemm)
  std::string quant_emb_path;
  std::string speech_emb_path;
  std::string sos_emb_path;
  std::string task_id_emb_path;
  std::string flow_input_emb_path;

  // Runtime settings
  int device_id = 0;        ///< Device ID for TCIM
  int sample_rate = 24000;  ///< Output audio sample rate

  /**
   * @brief Get full model path with default naming.
   * @param model_name Model name without extension.
   * @return Full path to .hmm file.
   */
  std::string GetModelPath(const std::string& model_name) const {
    if (!model_dir.empty()) {
      return model_dir + "/cosyvoice3_" + model_name + ".hmm";
    }
    return "";
  }

  /**
   * @brief Get full embedding path with default naming.
   * @param emb_name Embedding name without extension.
   * @return Full path to .hemm file.
   */
  std::string GetEmbeddingPath(const std::string& emb_name) const {
    if (!embedding_dir.empty()) {
      return embedding_dir + "/" + emb_name + ".bin";
    }
    return "";
  }
};

// ============================================================================
// Performance Metrics
// ============================================================================

/**
 * @brief Performance metrics for CosyVoice3 inference.
 * Reference: demo.py lines 91-131
 */
struct CosyVoice3Perf {
  // LLM metrics
  float llm_total_ms = 0.0f;  ///< Total LLM time
  float prefill_ms = 0.0f;    ///< Prefill phase time
  int prefill_tokens = 0;     ///< Prefill token count
  float ttft_ms = 0.0f;       ///< Time to first token
  float decode_ms = 0.0f;     ///< Decode phase time
  int decode_tokens = 0;      ///< Decode token count

  // Speaker embedding metrics
  float speaker_emb_ms = 0.0f;  ///< Speaker embedding time

  // Speech tokenizer metrics
  float speech_tokenizer_ms = 0.0f;  ///< Speech tokenizer time

  // Flow decoder metrics
  float flow_total_ms = 0.0f;    ///< Total Flow time
  float flow_encoder_ms = 0.0f;  ///< Flow encoder time
  float flow_decoder_ms = 0.0f;  ///< Flow decoder time

  // Vocoder metrics
  float vocoder_ms = 0.0f;  ///< HiFT vocoder time

  // End-to-end metrics
  float e2e_latency_s = 0.0f;     ///< Total latency in seconds
  float audio_duration_s = 0.0f;  ///< Output audio duration
  float rtf = 0.0f;               ///< Real-Time Factor (latency / duration)

  /**
   * @brief Print performance summary.
   */
  void Print() const {
    std::cout << "\n=== Performance Summary ===\n";
    std::cout << "LLM: " << llm_total_ms << " ms"
              << " (prefill: " << prefill_ms << " ms / " << prefill_tokens
              << " tokens"
              << ", decode: " << decode_ms << " ms / " << decode_tokens
              << " tokens"
              << ", TTFT: " << ttft_ms << " ms)\n";
    std::cout << "Speaker Emb: " << speaker_emb_ms << " ms\n";
    std::cout << "Speech Tokenizer: " << speech_tokenizer_ms << " ms\n";
    std::cout << "Flow: " << flow_total_ms << " ms"
              << " (encoder: " << flow_encoder_ms << " ms"
              << ", decoder: " << flow_decoder_ms << " ms)\n";
    std::cout << "Vocoder: " << vocoder_ms << " ms\n";
    std::cout << "E2E Latency: " << e2e_latency_s << " s\n";
    std::cout << "Audio Duration: " << audio_duration_s << " s\n";
    std::cout << "RTF: " << rtf << "\n";
    std::cout << "===========================\n\n";
  }
};

}  // namespace houmo

#endif  // COMMON_TYPES_H_