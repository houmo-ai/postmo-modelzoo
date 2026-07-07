/*
 * Copyright (c) 2026 HOUMO AI
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

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <map>
#include <random>
#include <regex>
#include <sstream>

#include "utils/HmEmbedding.h"
#include "utils/perf_tracker/inference_perf_tracker.h"
#include "utils/tcim_runtime_utils.h"

class HmllmInfer : public HmllmInferBase {
 public:
  /**
   * @brief Constructor for HmllmInfer class
   * @param prefillModelPath Path to the prefill model
   * @param decodeModelPath Path to the decode model
   * @param embeddingWeightPath Path to the embedding weights file
   * @param devices Vector of device IDs to use for inference
   * @param batches Batch size for the model
   * @param LazyMode Whether to use lazy mode for inference
   */
  HmllmInfer(const std::string& prefillModelPath,
             const std::string& decodeModelPath,
             const std::string& embeddingWeightPath,
             const std::vector<int>& devices, int batches, bool LazyMode);

  HmllmInfer(const HmllmInfer& it) = delete;

  HmllmInfer& operator=(const HmllmInfer& it) = delete;

  HmllmInfer(HmllmInfer&& it) noexcept = default;

  HmllmInfer& operator=(HmllmInfer&& it) noexcept = default;

  ~HmllmInfer();

  void DebugModelInfo(tcim::Module& module, const std::string& modelName);

  /**
   * @brief Performance test for LLM inference
   * @param input_tokens_len Number of input tokens
   * @param stop_tokens_len Number of tokens to generate before stopping
   * @return void
   */
  void perf_llm(const uint32_t input_tokens_len,
                const uint32_t stop_tokens_len) override;

  std::shared_ptr<InferencePerformanceTracker> get_perf_tracker() {
    return perf_tracker;
  }

 private:
  std::shared_ptr<InferencePerformanceTracker> perf_tracker;

  // Model paths
  std::string prefillModelPath = "";
  std::string decodeModelPath = "";

  // Configuration parameters - Model properties
  int prefill_length = 0;
  int embedding_length = 0;
  int64_t context_max_length = 0;
  int batch = 0;
  int argmax_dim_len = 0;
  int32_t decode_current_length = 1;
  std::shared_ptr<HmEmbedding> embedding;  // Embedding module

  tcim::Module::WeightManager weight_manager;    // Weight manager for model
  std::shared_ptr<tcim::Module> prefill_module;  // Prefill module instance
  std::shared_ptr<tcim::Module> decode_module;   // Decode module instance

  std::vector<std::string> dummy_names;  // Names for dummy tensors

  std::map<std::string, tcim::Tensor> prefill_input_map;
  std::map<std::string, tcim::Tensor> decode_input_map;

  uint32_t vocab_size = 0;
  // params for decode progress bar
  int bar_width = 50;
  int attn_idx_start = 0;

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

  void prefill_input_init();
  void decode_input_init();

  void PrefillSetInputDatas(void* data, int32_t valid_length,
                            int32_t current_length);

  /**
   * @brief Execute prefill inference
   */
  void PrefillInfer();

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
   */
  void DecodeInfer();

  /**
   * @brief Get token IDs from decode output
   * @param ids Vector to store the output token IDs
   */
  void DecodeGetOutputDatas(std::vector<int32_t>& ids);
};

#endif  // __LLMINFER_H__