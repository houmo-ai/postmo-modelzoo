/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: Hmtokenizer.h
 * Description:
 *   Tokenizer Interface Definition - Header file defining the HmTokenizer class for tokenization and embedding operations.
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

#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include "utils.h"

using half_float::half;
using tokenizers::Tokenizer;
using tensor_type = half;

/**
 * @brief HmTokenizer Class - Handles tokenization, detokenization, and token embedding operations
 *
 * This class provides functionalities for:
 * 1. Applying chat templates to message sequences
 * 2. Encoding text into token IDs
 * 3. Decoding token IDs back into text
 * 4. Generating embeddings from token IDs
 *
 * The tokenizer supports both step-by-step operations and one-step embedding generation from messages.
 */
class HmTokenizer {
public:
    /**
   * @brief Constructor - Initializes the tokenizer with required configurations and weights
   *
   * @param tokenizerJsonPath Path to the tokenizer.json file containing tokenizer configuration
   * @param embeddingWeightPath Path to the embedding_weight.pt file containing embedding weights
   * @param embedding_len Length of each embedding vector
   * @param prefill_len Maximum prefill length for token sequences
   */
    HmTokenizer(const std::string &tokenizerJsonPath,
                const std::string &embeddingWeightPath, const int &embedding_len,
                const int &prefill_len);

    /**
   * @brief Copy constructor - Disabled (no copy allowed)
   */
    HmTokenizer(const HmTokenizer &it) = delete;

    /**
   * @brief Copy assignment operator - Disabled (no copy allowed)
   */
    HmTokenizer &operator=(const HmTokenizer &it) = delete;

    /**
   * @brief Move constructor - Default (move allowed)
   */
    HmTokenizer(HmTokenizer &&it) noexcept = default;

    /**
   * @brief Move assignment operator - Default (move allowed)
   */
    HmTokenizer &operator=(HmTokenizer &&it) noexcept = default;

    /**
   * @brief Destructor - Cleans up allocated resources
   */
    ~HmTokenizer();

    /**
   * @brief Applies a chat template to a sequence of messages
   *
   * Formats messages into a text string suitable for model input according to the chat template.
   *
   * @param msgs Vector of Message objects containing chat history
   * @param add_generation_prompt Whether to add a prompt for generation after the messages
   * @param enable_thinking Whether to enable thinking mode in the chat template
   * @return Formatted text string ready for encoding
   */
    std::string ApplyChatTemplate(const std::vector<Message> &msgs,
                                  bool add_generation_prompt = true,
                                  bool enable_thinking = false);

    /**
   * @brief Encodes text into token IDs
   *
   * Converts a formatted text string into a sequence of token IDs using the tokenizer.
   *
   * @param text Formatted text string to encode
   * @return Vector of token IDs
   */
    std::vector<int> Encode(const std::string &text);

    /**
   * @brief Decodes token IDs back into text
   *
   * Converts a sequence of token IDs into human-readable text.
   *
   * @param ids Vector of token IDs to decode
   * @return Decoded human-readable text
   */
    std::string Decode(const std::vector<int32_t> &ids);

    /**
   * @brief Generates embeddings from token IDs
   *
   * Converts a sequence of token IDs into embedding vectors using preloaded embedding weights.
   *
   * @param ids Vector of token IDs
   * @return Pointer to tensor_type (half precision) containing embedding vectors
   */
    tensor_type *EmbeddingTokens(const std::vector<int> &ids);

    /**
   * @brief One-step embedding generation from messages
   *
   * Combines ApplyChatTemplate, Encode, and EmbeddingTokens into a single operation.
   *
   * @param msgs Vector of Message objects containing chat history
   * @param add_generation_prompt Whether to add a prompt for generation after the messages
   * @param enable_thinking Whether to enable thinking mode in the chat template
   * @return Pointer to tensor_type (half precision) containing embedding vectors
   */
    tensor_type *EmbeddingTokens(const std::vector<Message> &msgs,
                                 bool add_generation_prompt = true,
                                 bool enable_thinking = false);

private:
    std::unique_ptr<Tokenizer> tok;          // Unique pointer to the underlying tokenizer instance
    std::unique_ptr<tensor_type[]> embed_w;  // Unique pointer to the embedding weight matrix

    tensor_type *ptr = nullptr;  // Pointer to allocated memory for embeddings
    size_t ptr_size = 0;         // Size of the allocated memory for embeddings

    int prefill_length = 0;    // Maximum prefill length for token sequences
    int embedding_length = 0;  // Length of each embedding vector
};

#endif  // __TOKENIZER_H__