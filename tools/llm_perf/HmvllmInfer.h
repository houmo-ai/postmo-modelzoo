/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmvllmInfer.h
 * Description:
 *   HmvllmInfer Header File - Defines the HmvllmInfer class for vision-language
 * large language model inference performance testing.
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
#ifndef __VLLMINFER_H__
#define __VLLMINFER_H__

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>

#include "HmEmbedding.h"
#include "tcim_runtime_utils.h"
#include "utils.h"

/**
 * @brief Vision-Language Large Language Model Inference class
 * Handles performance testing for vision-language models that combine
 * visual processing with text generation capabilities
 */
class HmvllmInfer : public HmllmInferBase {
 public:
  /**
   * @brief Constructor for HmvllmInfer
   * @param prefillModelPath Path to the prefill model
   * @param decodeModelPath Path to the decode model
   * @param embeddingWeightPath Path to the embedding weights
   * @param visionModelPath Path to the Vision Transformer model
   * @param devices Vector of device IDs to use for inference
   * @param batches Number of batches for processing
   * @param LazyMode Whether to use lazy mode for inference
   */
  HmvllmInfer(const std::string &prefillModelPath,
              const std::string &decodeModelPath,
              const std::string &embeddingWeightPath,
              const std::string &visionModelPath,
              const std::vector<int> &devices, int batches, bool LazyMode);

  HmvllmInfer(const HmvllmInfer &it) = delete;

  HmvllmInfer &operator=(const HmvllmInfer &it) = delete;

  HmvllmInfer(HmvllmInfer &&it) noexcept = default;

  HmvllmInfer &operator=(HmvllmInfer &&it) noexcept = default;

  ~HmvllmInfer();

  /**
   * @brief Debug function to print model information
   * @param module Reference to the module to debug
   * @param modelName Name of the model to display in debug output
   */
  void DebugModelInfo(tcim::Module &module, const std::string &modelName);

  /**
   * @brief Performance test for vision-language model inference
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
  std::string visionModelPath = "";

  // Configuration parameters
  int prefill_length = 0;
  int embedding_length = 0;
  int context_max_length = 0;
  int batch = 0;
  int argmax_dim_len = 0;  // Dimension length for argmax operation

  std::shared_ptr<HmEmbedding> embedding;

  tcim::Module::WeightManager weight_manager;         // LLM weight manager
  tcim::Module::WeightManager vision_weight_manager;  // Vision weight manager
  // module instance for all vision models.
  std::shared_ptr<tcim::Module> prefill_module;
  std::shared_ptr<tcim::Module> decode_module;
  std::shared_ptr<tcim::Module> vision_module;

  // Names for dummy tensors in model
  std::vector<std::string> dummy_names;

  // Input tensor maps for different models.
  std::map<std::string, tcim::Tensor> prefill_input_map;
  std::map<std::string, tcim::Tensor> decode_input_map;
  std::map<std::string, tcim::Tensor> vision_input_map;

  int bar_width = 50;      // Width for progress bar display
  int attn_idx_start = 0;  // Starting index for attention inputs
  int vision_input_nums = 0;
  int past_seq_len = 0;
  uint32_t vocab_size = 0;

  bool prefill_use_vision_outputs = false;

  std::map<std::string, std::unique_ptr<char[]>> prefill_input_datas;
  std::map<std::string, std::unique_ptr<char[]>> decode_input_datas;
  std::map<std::string, std::unique_ptr<char[]>> vision_input_datas;

 private:
  void prefill_input_init();
  void decode_input_init();
  void vision_input_init();
  /**
   * @brief Get the number of transformer blocks in the model
   * @return Number of blocks in the model
   */
  int get_nblocks();

  /**
   * @brief Get the starting index of attention-related inputs
   * @return Starting index for attention inputs
   */
  int get_attn_idx_start();

  /**
   * @brief Set input data for prefill operation
   * @param data Input data for the prefill stage
   */
  void PrefillSetInputDatas(void *data, int current_length);

  /**
   * @brief Execute prefill inference
   * @return void
   */
  void PrefillInfer();

  /**
   * @brief Get output token IDs from prefill stage
   * @param ids Vector to store the output token IDs
   */
  void PrefillGetOutputDatas(std::vector<int32_t> &ids);

  /**
   * @brief Set input data for decode operation
   * @param data Input data for the decode stage
   * @param valid_length Length of valid tokens in the sequence
   */
  void DecodeSetInputDatas(void *data, int valid_length);

  /**
   * @brief Execute decode inference
   * @return void
   */
  void DecodeInfer();

  /**
   * @brief Get output token IDs from decode stage
   * @param ids Vector to store the output token IDs
   */
  void DecodeGetOutputDatas(std::vector<int32_t> &ids);

  /**
   * @brief Set input data for Vision Model
   */
  void VisionSetInput();

  /**
   * @brief Execute Vision Model inference
   * @return void
   */
  void VisionInfer();

  /**
   * @brief Get output data from Vision Model
   */
  void VisionGetOutputDatas();
};

#endif  // __VLLMINFER_H__