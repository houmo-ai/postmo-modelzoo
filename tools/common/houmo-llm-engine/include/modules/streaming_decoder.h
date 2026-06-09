/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: streaming_decoder.h
 * Description:
 *   Streaming decoder using sliding window to handle UTF-8 multi-byte
 *   character boundaries during token-by-token generation.
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

#include "base/houmo.h"
#include "modules/tokenizer.h"

#include <memory>
#include <string>
#include <vector>

namespace houmo {

/**
 * @brief Streaming decoder
 *
 * Converts token stream to string stream output. Uses sliding window
 * mechanism to handle UTF-8 multi-byte character boundaries.
 *
 * Usage:
 *   StreamingDecoder decoder(model.tokenizer());
 *   ctx->generate(tokens, params, [&](Token token) {
 *       std::cout << decoder.decode(token);
 *       std::cout.flush();
 *       return true;
 *   });
 */
class StreamingDecoder {
 public:
  /**
   * @brief Constructor
   * @param tokenizer Shared HfTokenizer pointer (thread-safe, can be shared)
   */
  explicit StreamingDecoder(std::shared_ptr<HfTokenizer> tokenizer);

  ~StreamingDecoder() = default;

  // Non-copyable
  StreamingDecoder(const StreamingDecoder&) = delete;
  StreamingDecoder& operator=(const StreamingDecoder&) = delete;

  // Movable
  StreamingDecoder(StreamingDecoder&&) noexcept = default;
  StreamingDecoder& operator=(StreamingDecoder&&) noexcept = default;

  /**
   * @brief Decode a single token
   * @param token Newly generated token
   * @return Newly decoded string portion (may be empty)
   *
   * Uses sliding window to ensure complete UTF-8 multi-byte character output.
   * Returns empty string and caches when encountering incomplete UTF-8
   * characters, outputting them after subsequent tokens complete them.
   */
  std::string decode(Token token);

  /**
   * @brief Initialize decoder with prompt tokens
   * @param tokens Prompt tokens
   *
   * In manual Prefill+Decode mode, call this before decode() to
   * initialize the sliding window baseline.
   */
  void init(const std::vector<Token>& tokens);

  /**
   * @brief Clear decoder state
   *
   * Call before starting a new generation session.
   */
  void reset();

  /**
   * @brief Get number of generated tokens
   */
  size_t token_count() const { return generated_ids_.size(); }

 private:
  /**
   * @brief Check if Unicode code point is a valid character (CJK, ASCII letter or digit)
   */
  static bool is_valid_char(char32_t cp);

  std::shared_ptr<HfTokenizer> tokenizer_;
  std::vector<Token> generated_ids_;
  std::string last_response_;
  int skip_tokens_ = 0;
  static constexpr int kSlideLen = 10;  // Sliding window length
};

}  // namespace houmo
