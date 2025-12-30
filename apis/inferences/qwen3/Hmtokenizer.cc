/*
 * Hmtokenizer.h - HoumoAI Tokenizer Implementation for Text Processing and Embedding
 *
 * Copyright (c) 2025 HOUMOAI
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
#include "Hmtokenizer.h"

/**
 * @brief Constructor for HmTokenizer. Initializes tokenizer and embedding weights.
 *
 * @param tokenizerJsonPath Path to tokenizer.json file
 * @param embeddingWeightPath Path to embedding.bin file
 * @param embedding_len Length of each embedding vector
 * @param prefill_len Maximum number of tokens for prefill
 */
HmTokenizer::HmTokenizer(const std::string &tokenizerJsonPath,
                         const std::string &embeddingWeightPath,
                         const int &embedding_len,
                         const int &prefill_len) : prefill_length(prefill_len),
                                                   embedding_length(embedding_len) {
    // Load tokenizer configuration from JSON file
    auto blob = LoadBytesFromFile(tokenizerJsonPath);
    tok = Tokenizer::FromBlobJSON(blob);

    // Load embedding weights; allocate additional 1MB space for direct pointer return during decode
    embed_w = readEmbeddingWeight<tensor_type>(embeddingWeightPath, prefill_length * embedding_length);

    // Allocate buffer for storing embeddings of multiple tokens
    ptr = new tensor_type[prefill_length * embedding_length];
}

/**
 * @brief Destructor for HmTokenizer. Cleans up resources.
 */
HmTokenizer::~HmTokenizer() {
    // Release tokenizer resources
    tok.reset();

    // Free the allocated embedding buffer
    delete ptr;
}

/**
 * @brief Applies chat template to format messages into model-compatible text.
 *
 * @param msgs Vector of Message objects containing role and content
 * @param add_generation_prompt Whether to add assistant prompt at the end
 * @param enable_thinking Whether to enable thinking mode (controls special tokens)
 * @return Formatted string according to chat template
 */
std::string HmTokenizer::ApplyChatTemplate(const std::vector<Message> &msgs,
                                           bool add_generation_prompt,
                                           bool enable_thinking) {
    std::string out;
    out.reserve(1024);  // Pre-allocate buffer for efficiency

    // Process each message in the vector
    for (const auto &m : msgs) {
        out.append("<|im_start|>");
        out.append(m.role);
        out.push_back('\n');
        out.append(m.content);
        out.append("<|im_end|>\n");
    }

    // Add assistant prompt if requested
    if (add_generation_prompt) {
        out.append("<|im_start|>assistant\n");
    }

    // Add special thinking mode tokens if disabled
    if (!enable_thinking) {
        out.append("\n");
        out.append("\n");
        out.append("\n");
        out.append("\n");
    }

    return out;
}

/**
 * @brief Encodes text into a sequence of token IDs.
 *
 * @param text Input text to encode
 * @return Vector of token IDs
 */
std::vector<int> HmTokenizer::Encode(const std::string &text) {
    std::vector<int> ids = tok->Encode(text);
    return ids;
}

/**
 * @brief Decodes token IDs back into text.
 *
 * @param ids Vector of token IDs to decode
 * @return Decoded text string
 */
std::string HmTokenizer::Decode(const std::vector<int32_t> &ids) {
    return tok->Decode(ids);
}

/**
 * @brief Retrieves embeddings for a sequence of token IDs.
 *
 * @param ids Vector of token IDs to get embeddings for
 * @return Pointer to embedding tensor (either direct weight pointer for single token or copied buffer for multiple tokens)
 */
tensor_type *HmTokenizer::EmbeddingTokens(const std::vector<int> &ids) {
    uint64_t num_tokens = ids.size();

    if (num_tokens == 0) {
        return nullptr;  // Return null if no tokens provided
    }

    if (num_tokens == 1) {
        // Return direct pointer to embedding weight for single token (no copy needed)
        return reinterpret_cast<tensor_type *>(&embed_w[ids[0] * embedding_length]);
    }

    // Clear the pre-allocated buffer
    memset(reinterpret_cast<void *>(ptr), 0, prefill_length * embedding_length * sizeof(tensor_type));

    // Copy embeddings from weight matrix to buffer for multiple tokens
    for (int index = 0; index < num_tokens; index++) {
        int embedWeightIndx = ids[index];
        memcpy(reinterpret_cast<void *>(&ptr[index * embedding_length]),
               reinterpret_cast<void *>(&embed_w[embedWeightIndx * embedding_length]),
               embedding_length * sizeof(tensor_type));
    }

    return ptr;
}

/**
 * @brief Retrieves embeddings for messages formatted with chat template.
 *
 * @param msgs Vector of Message objects
 * @param add_generation_prompt Whether to add assistant prompt
 * @param enable_thinking Whether to enable thinking mode
 * @return Pointer to embedding tensor
 */
tensor_type *HmTokenizer::EmbeddingTokens(const std::vector<Message> &msgs,
                                          bool add_generation_prompt,
                                          bool enable_thinking) {
    if (msgs.empty()) {
        return nullptr;  // Return null if no messages provided
    }

    // Render messages into formatted text using chat template
    std::string rendered = ApplyChatTemplate(msgs, add_generation_prompt, enable_thinking);

    // Encode formatted text into token IDs
    std::vector<int> ids = Encode(rendered);

    // Get embeddings for token IDs
    return EmbeddingTokens(ids);
}