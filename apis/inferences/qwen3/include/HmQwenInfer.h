/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmQwenInfer.h
 * Description:
 *   Header file for Qwen inference class using tcim runtime.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef __HMQWENINFER_H__
#define __HMQWENINFER_H__

#include <iomanip>
#include <map>
#include <regex>

#include "Hmtokenizer.h"
#include "tcim/tcim_runtime.h"
#include "utils.h"

/**
 * @struct PerfInfos
 * @brief Stores performance metrics for model inference
 */
struct PerfInfos {
    int input_tokens;
    int output_tokens;
    float ttft_time;
    float prefill_time;    // Time taken for prefill phase (in seconds)
    float decode_time;     // Time taken for decode phase (in seconds)
    float embedding_time;  // Time taken for embedding generation (in seconds)
};

/**
 * @class HmQwenInfer
 * @brief Main class for Qwen model inference
 */
class HmQwenInfer {
public:
    /**
   * @brief Constructor for HmQwenInfer
   * @param prefillModelPath Path to prefill model
   * @param decodeModelPath Path to decode model
   * @param tokenizerJsonPath Path to tokenizer JSON configuration
   * @param embeddingWeightPath Path to embedding weights
   */
    HmQwenInfer(const std::string &prefillModelPath,
                const std::string &decodeModelPath,
                const std::string &tokenizerJsonPath,
                const std::string &embeddingWeightPath);
    HmQwenInfer(const HmQwenInfer &it) = delete;                  // Delete copy constructor
    HmQwenInfer &operator=(const HmQwenInfer &it) = delete;       // Delete copy assignment
    HmQwenInfer(HmQwenInfer &&it) noexcept = default;             // Default move constructor
    HmQwenInfer &operator=(HmQwenInfer &&it) noexcept = default;  // Default move assignment
    ~HmQwenInfer();                                               // Destructor

    /**
   * @brief Gets the model module based on type
   * @param model_type Type of model: 0-prefill, 1-decode
   * @return Shared pointer to the module, or nullptr if input is invalid
   */
    std::shared_ptr<tcim::Module> GetModule(int model_type);

    /**
   * @brief Gets the tokenizer instance
   * @return Shared pointer to the tokenizer
   */
    std::shared_ptr<HmTokenizer> GetTokenizer();

    /**
   * @brief Prints model input and output information
   * @param module Model module
   * @param modelName Name of the model
   */
    void DebugModelInfo(tcim::Module &module, const std::string &modelName);

    /**
   * @brief Processes user input and generates response
   * @param msg User input message
   */
    void Chat(const std::string &msg);

private:
    // Model paths
    std::string prefillModelPath = "";  // Path to prefill model
    std::string decodeModelPath = "";   // Path to decode model

    // Configuration parameters
    int prefill_length = 0;             // Maximum prefill length
    int embedding_length = 0;           // Embedding dimension
    int context_max_length = 0;         // Maximum context length
    int batch = 0;                      // Batch size
    int eos_token_id = 0;               // End-of-sequence token ID
    int argmax_dim_len = 0;             // Dimension length for argmax operation
    int32_t decode_current_length = 1;  // Current decode length

    // Model related members
    std::shared_ptr<HmTokenizer> tokenizer;  // Tokenizer instance

    tcim::Module::WeightManager weight_manager;    // Weight manager
    std::shared_ptr<tcim::Module> prefill_module;  // Prefill model module
    std::shared_ptr<tcim::Module> decode_module;   // Decode model module

    std::vector<std::string> dummy_names;  // Dummy names for tensor handling

    // Tensor maps for input/output
    std::map<std::string, tcim::Tensor> prefill_input_map;   // Prefill input tensors
    std::map<std::string, tcim::Tensor> decode_input_map;    // Decode input tensors
    std::map<std::string, tcim::Tensor> prefill_output_map;  // Prefill output tensors
    std::map<std::string, tcim::Tensor> decode_output_map;   // Decode output tensors

private:
    /**
   * @brief Gets the number of blocks in prefill model
   * @return Number of blocks
   */
    int GetnBlocks();

    /**
   * @brief Sets input data for prefill phase
   * @param data Input data pointer
   * @param valid_length Valid input length
   * @param current_length Current input length
   */
    void PrefillSetInputDatas(void *data, int32_t valid_length,
                              int32_t current_length);

    /**
   * @brief Performs prefill model inference
   */
    void PrefillInfer();

    /**
   * @brief Gets output token IDs from prefill phase
   * @param ids Vector to store output token IDs
   */
    void PrefillGetOutputDatas(std::vector<int32_t> &ids);

    /**
   * @brief Sets input data for decode phase
   * @param data Input data pointer
   * @param context_length Context length
   */
    void DecodeSetInputDatas(void *data, int32_t context_length);

    /**
   * @brief Performs decode model inference
   */
    void DecodeInfer();

    /**
   * @brief Gets output token IDs from decode phase
   * @param ids Vector to store output token IDs
   */
    void DecodeGetOutputDatas(std::vector<int32_t> &ids);
};

#endif  // __HMQWENINFER_H__