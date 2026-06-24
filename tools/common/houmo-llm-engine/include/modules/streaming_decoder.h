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
 * Converts token stream to string stream output. Buffers pending tokens
 * and decodes them together to handle UTF-8 multi-byte character boundaries
 * where a single character (e.g. emoji or rare CJK) may span multiple tokens.
 *
 * On each new token:
 *   1. Add to pending buffer
 *   2. Decode the entire buffer
 *   3. If result is valid UTF-8 with a valid trailing character -> output and clear buffer
 *   4. If result is incomplete (invalid UTF-8 or trailing replacement char) -> keep buffering
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
   * Buffers tokens until a complete, valid UTF-8 string can be produced.
   * Returns empty string when the buffer contains incomplete multi-byte
   * characters, outputting them after subsequent tokens complete them.
   */
  std::string decode(Token token);

  /**
   * @brief Initialize decoder with prompt tokens
   * @param tokens Prompt tokens
   *
   * In manual Prefill+Decode mode, call this before decode() to
   * initialize the token counter.
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
  static bool is_valid_utf8(const std::string& s);
  static bool is_valid_char(char32_t cp);
  static char32_t last_codepoint(const std::string& s);

  std::shared_ptr<HfTokenizer> tokenizer_;
  std::vector<Token> generated_ids_;
  std::vector<Token> pending_ids_;
};

}  // namespace houmo
