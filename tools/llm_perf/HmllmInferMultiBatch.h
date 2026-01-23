/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmllmInferMultiBatch.h
 * Description:
 *   HmllmInferMultiBatch Header File - Defines the HmllmInferMultiBatch class
 * for multi-batch large language model inference performance testing.
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
#ifndef __HMLLMINFERMULTIBATCH_H__
#define __HMLLMINFERMULTIBATCH_H__

#include <chrono>
#include <iomanip>
#include <map>
#include <random>
#include <regex>
#include <sstream>

#include "HmEmbedding.h"
#include "tcim/tcim_runtime.h"

/**
 * @brief Structure to hold performance information for a single batch
 */
struct PerfSingleBatchInfo {
  uint32_t input_tokens;     // Number of input tokens processed
  float time;                // Execution time for the batch
  float embedding_time;      // Time spent on embedding operations
  std::vector<int> next_id;  // Next token IDs generated
};

/**
 * @brief Class for multi-batch large language model inference performance
 * testing
 */
class HmllmInferMultiBatch : public HmllmInferBase {
 public:
  /**
   * @brief Constructor for HmllmInferMultiBatch
   * @param prefillModelPath Path to the prefill model
   * @param decodeModelPath Path to the decode model
   * @param embeddingWeightPath Path to the embedding weights
   * @param ndevices Number of devices to use
   * @param batches Number of batches for multi-batch processing
   */
  HmllmInferMultiBatch(const std::string &prefillModelPath,
                       const std::string &decodeModelPath,
                       const std::string &embeddingWeightPath, int ndevices,
                       int batches, bool LazyMode);

  // Delete copy constructor to prevent copying
  HmllmInferMultiBatch(const HmllmInferMultiBatch &it) = delete;

  // Delete assignment operator to prevent assignment
  HmllmInferMultiBatch &operator=(const HmllmInferMultiBatch &it) = delete;

  // Default move constructor
  HmllmInferMultiBatch(HmllmInferMultiBatch &&it) noexcept = default;

  // Default move assignment operator
  HmllmInferMultiBatch &operator=(HmllmInferMultiBatch &&it) noexcept = default;

  // Destructor
  ~HmllmInferMultiBatch();

  /**
   * @brief Debug function to print model information
   * @param module Module to debug
   * @param modelName Name of the model for display
   */
  void DebugModelInfo(tcim::Module &module, const std::string &modelName);

  /**
   * @brief Perform LLM performance test with specified token lengths
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
  std::string prefillModelPath = "";  // Path to prefill model
  std::string decodeModelPath = "";   // Path to decode model

  // Configuration parameters - Model properties
  int prefill_length = 0;      // Length of prefill operation
  int embedding_length = 0;    // Length of embedding vectors
  int context_max_length = 0;  // Maximum context length supported
  int batch = 0;               // Batch size
  int eos_token_id = 0;        // End-of-sequence token ID
  int argmax_dim_len = 0;      // Dimension length for argmax operation

  int32_t decode_current_length = 1;  // Current length for decode operation

  std::shared_ptr<HmEmbedding> embedding;  // Embedding module instance

  tcim::Module::WeightManager weight_manager;    // Weight manager for models
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

  std::vector<std::vector<int>> next_ids;  // Next token IDs for each batch
  std::vector<int> current_echo_lens;  // Current echo lengths for each batch

  int bar_width = 50;  // Width for progress bar display
  int n_blocks = 0;    // Number of transformer blocks

 private:
  /**
   * @brief Get the number of blocks in the model
   * @return Number of transformer blocks in the model
   */
  int get_nblocks();

  /**
   * @brief Set prefill input data
   * @param data Input data for prefill operation
   * @param valid_length Valid length of the input sequence
   * @param current_length Current length of the sequence
   */
  void PrefillSetInputDatas(void *data, int32_t valid_length,
                            int32_t current_length);

  /**
   * @brief Execute prefill inference
   */
  void PrefillInfer();

  /**
   * @brief Get token IDs from prefill output
   * @param ids Vector to store the output token IDs
   */
  void PrefillGetOutputDatas(std::vector<int32_t> &ids);

  /**
   * @brief Run prefill operation for a specific batch
   * @param batch Batch index to process
   * @param all_input_ids Input token IDs for the batch
   * @return Performance information for the single batch
   */
  PerfSingleBatchInfo run_prefill(int batch,
                                  const std::vector<int> all_input_ids);

  /**
   * @brief Run decode operation with given input data and context lengths
   * @param input_datas Input embedding data
   * @param context_length Context lengths for each batch
   * @return Performance information for the single batch
   */
  PerfSingleBatchInfo run_decode(tensor_type *input_datas,
                                 const std::vector<int> context_length);

  /**
   * @brief Set decode input data
   * @param data Input data for decode operation
   * @param context_length Context length for the sequence
   */
  void DecodeSetInputDatas(void *data, int32_t context_length);

  /**
   * @brief Execute decode inference
   */
  void DecodeInfer();

  /**
   * @brief Get token IDs from decode output for all batches
   */
  void DecodeGetOutputDatas();
};

#endif  // __HMLLMINFERMULTIBATCH_H__