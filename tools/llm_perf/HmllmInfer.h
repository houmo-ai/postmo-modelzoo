/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmllmInfer.h
 * Description:
 *   HmllmInfer Header File - Defines the HmllmInfer class for large language
 * model inference performance testing.
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
#ifndef __LLMINFER_H__
#define __LLMINFER_H__

#include <chrono>
#include <iomanip>
#include <map>
#include <random>
#include <regex>
#include <sstream>

#include "HmEmbedding.h"
#include "tcim/tcim_runtime.h"

class HmllmInfer : public HmllmInferBase {
 public:
  /**
   * @brief Constructor for HmllmInfer class
   * @param prefillModelPath Path to the prefill model
   * @param decodeModelPath Path to the decode model
   * @param embeddingWeightPath Path to the embedding weights file
   * @param ndevices Number of devices to use for inference
   * @param batches Batch size for the model
   */
  HmllmInfer(const std::string& prefillModelPath,
             const std::string& decodeModelPath,
             const std::string& embeddingWeightPath, int ndevices, int batches);

  // Delete copy constructor to prevent copying of the object
  HmllmInfer(const HmllmInfer& it) = delete;

  // Delete assignment operator to prevent assignment of the object
  HmllmInfer& operator=(const HmllmInfer& it) = delete;

  // Default move constructor for efficient object transfer
  HmllmInfer(HmllmInfer&& it) noexcept = default;

  // Default move assignment operator for efficient object transfer
  HmllmInfer& operator=(HmllmInfer&& it) noexcept = default;

  // Destructor for cleaning up resources
  ~HmllmInfer();

  /**
   * @brief Debug function to print model information
   * @param module Reference to the module to debug
   * @param modelName Name of the model to display in debug output
   */
  void DebugModelInfo(tcim::Module& module, const std::string& modelName);

  /**
   * @brief Performance test for LLM inference
   * @param input_tokens_len Number of input tokens
   * @param stop_tokens_len Number of tokens to generate before stopping
   * @return Performance information structure
   */
  PerfInfos perf_llm(const uint32_t input_tokens_len,
                     const uint32_t stop_tokens_len) override;

 private:
  // Model paths
  std::string prefillModelPath = "";  // Path to the prefill model
  std::string decodeModelPath = "";   // Path to the decode model

  // Configuration parameters - Model properties
  int prefill_length = 0;             // Length of prefill operation
  int embedding_length = 0;           // Length of embedding vectors
  int context_max_length = 0;         // Maximum context length supported
  int batch = 0;                      // Batch size
  int eos_token_id = 0;               // End-of-sequence token ID
  int argmax_dim_len = 0;             // Dimension length for argmax operation
  int32_t decode_current_length = 1;  // Current length for decode operation
  std::shared_ptr<HmEmbedding> embedding;  // Embedding module

  tcim::Module::WeightManager weight_manager;    // Weight manager for model
  std::shared_ptr<tcim::Module> prefill_module;  // Prefill module instance
  std::shared_ptr<tcim::Module> decode_module;   // Decode module instance

  std::vector<std::string> dummy_names;  // Names for dummy tensors

  std::map<std::string, tcim::Tensor>
      prefill_input_map;  // Prefill input tensor mapping
  std::map<std::string, tcim::Tensor>
      decode_input_map;  // Decode input tensor mapping
  std::map<std::string, tcim::Tensor>
      prefill_output_map;  // Prefill output tensor mapping
  std::map<std::string, tcim::Tensor>
      decode_output_map;  // Decode output tensor mapping

  int bar_width = 50;      // Width for progress bar display
  int attn_idx_start = 0;  // Starting index for attention inputs

 private:
  /**
   * @brief Get the number of blocks in the model
   * @return Number of blocks in the model
   */
  int get_nblocks();

  /**
   * @brief Get the starting index of attention inputs
   * @return Starting index of attention-related inputs
   */
  int get_attn_idx_start();

  /**
   * @brief Set prefill input data
   * @param data Input data for prefill (token embeddings)
   * @param valid_length Length of valid tokens in the input
   * @param current_length Current length of the sequence
   */
  void PrefillSetInputDatas(void* data, int32_t valid_length,
                            int32_t current_length);

  /**
   * @brief Execute prefill inference
   * @return Execution time in milliseconds
   */
  float PrefillInfer();

  /**
   * @brief Get token IDs from prefill output
   * @param ids Vector to store the output token IDs
   */
  void PrefillGetOutputDatas(std::vector<int32_t>& ids);

  /**
   * @brief Set decode input data
   * @param data Input data for decode (token embeddings)
   * @param context_length Current context length
   */
  void DecodeSetInputDatas(void* data, int32_t context_length);

  /**
   * @brief Execute decode inference
   * @return Execution time in milliseconds
   */
  float DecodeInfer();

  /**
   * @brief Get token IDs from decode output
   * @param ids Vector to store the output token IDs
   */
  void DecodeGetOutputDatas(std::vector<int32_t>& ids);
};

#endif  // __LLMINFER_H__