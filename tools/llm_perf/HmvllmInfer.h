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
#include "tcim/tcim_runtime.h"
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
   * @param vitModelPath Path to the Vision Transformer (ViT) model
   * @param ndevices Number of devices to use for inference
   * @param batches Number of batches for processing
   */
  HmvllmInfer(const std::string &prefillModelPath,
              const std::string &decodeModelPath,
              const std::string &embeddingWeightPath,
              const std::string &vitModelPath, int ndevices, int batches);

  // Delete copy constructor to prevent copying of the object
  HmvllmInfer(const HmvllmInfer &it) = delete;

  // Delete assignment operator to prevent assignment of the object
  HmvllmInfer &operator=(const HmvllmInfer &it) = delete;

  // Default move constructor for efficient object transfer
  HmvllmInfer(HmvllmInfer &&it) noexcept = default;

  // Default move assignment operator for efficient object transfer
  HmvllmInfer &operator=(HmvllmInfer &&it) noexcept = default;

  // Destructor for resource cleanup
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
   * @return Performance information structure
   */
  PerfInfos perf_llm(const uint32_t input_tokens_len,
                     const uint32_t stop_tokens_len) override;

 private:
  // Model paths
  std::string prefillModelPath = "";  // Path to the prefill model
  std::string decodeModelPath = "";   // Path to the decode model
  std::string vitModelPath = "";      // Path to the Vision Transformer model

  // Configuration parameters
  int prefill_length = 0;      // Length of prefill operation
  int embedding_length = 0;    // Length of embedding vectors
  int context_max_length = 0;  // Maximum context length supported
  int batch = 0;               // Batch size
  int argmax_dim_len = 0;      // Dimension length for argmax operation

  std::shared_ptr<HmEmbedding>
      embedding;  // Embedding module for token processing

  tcim::Module::WeightManager weight_manager;    // Weight manager for models
  std::shared_ptr<tcim::Module> prefill_module;  // Prefill module instance
  std::shared_ptr<tcim::Module> decode_module;   // Decode module instance
  std::shared_ptr<tcim::Module>
      vit_module;  // Vision Transformer module instance

  std::vector<std::string> dummy_names;  // Names for dummy tensors in model

  // Input tensor maps for different modules
  std::map<std::string, tcim::Tensor>
      prefill_input_map;  // Prefill input tensor mapping
  std::map<std::string, tcim::Tensor>
      decode_input_map;  // Decode input tensor mapping
  std::map<std::string, tcim::Tensor>
      vit_input_map;  // ViT input tensor mapping

  // Output tensor maps for different modules
  std::map<std::string, tcim::Tensor>
      prefill_output_map;  // Prefill output tensor mapping
  std::map<std::string, tcim::Tensor>
      decode_output_map;  // Decode output tensor mapping
  std::map<std::string, tcim::Tensor>
      vit_output_map;  // ViT output tensor mapping

  int bar_width = 50;      // Width for progress bar display
  int attn_idx_start = 0;  // Starting index for attention inputs
  int vit_input_nums = 0;  // Number of inputs for ViT module

  // Pointers to input data for different modules
  std::vector<char *> prefill_input_ptrs;  // Pointers to prefill input data
  std::vector<char *> decode_input_ptrs;   // Pointers to decode input data
  std::vector<char *> vit_input_ptrs;      // Pointers to ViT input data

 private:
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
  void PrefillSetInputDatas(void *data);

  /**
   * @brief Execute prefill inference and return execution time
   * @return Execution time in milliseconds
   */
  float PrefillInfer();

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
   * @brief Execute decode inference and return execution time
   * @return Execution time in milliseconds
   */
  float DecodeInfer();

  /**
   * @brief Get output token IDs from decode stage
   * @param ids Vector to store the output token IDs
   */
  void DecodeGetOutputDatas(std::vector<int32_t> &ids);

  /**
   * @brief Set input data for Vision Transformer (ViT) operation
   */
  void VitSetInput();

  /**
   * @brief Execute Vision Transformer inference and return execution time
   * @return Execution time in milliseconds
   */
  float VitInfer();

  /**
   * @brief Get output data from Vision Transformer stage
   */
  void VitGetOutputDatas();
};

#endif  // __VLLMINFER_H__