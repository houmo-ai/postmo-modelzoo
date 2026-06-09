/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: streaming_decoder.cc
 * Description:
 *   Streaming decoder implementation for incremental token-to-text decoding.
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

#include "modules/streaming_decoder.h"

#include <algorithm>
#include <codecvt>
#include <locale>

namespace houmo {

// UTF-8 helper functions
static std::u32string utf8_to_u32(const std::string& u8) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.from_bytes(u8);
}

static std::string u32_to_utf8(const std::u32string& u32) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.to_bytes(u32);
}

static size_t utf8_len(const std::string& u8) { return utf8_to_u32(u8).size(); }

// ============================================================================
// StreamingDecoder implementation
// ============================================================================

StreamingDecoder::StreamingDecoder(std::shared_ptr<HfTokenizer> tokenizer)
    : tokenizer_(tokenizer) {}

std::string StreamingDecoder::decode(Token token) {
  if (!tokenizer_) {
    return "";
  }

  // Add new token to history
  generated_ids_.push_back(token);

  // Calculate decode window: last slide_len + skip_tokens + 1 tokens
  int window_size = kSlideLen + skip_tokens_ + 1;
  int start_idx =
      std::max(0, static_cast<int>(generated_ids_.size()) - window_size);

  std::vector<Token> decode_window(generated_ids_.begin() + start_idx,
                                   generated_ids_.end());

  // Decode window
  std::string decoded = tokenizer_->decode(decode_window);

  // Calculate UTF-8 character count of last_response_ as start position
  int substart = utf8_len(last_response_);

  // Convert to UTF-32 and extract incremental part
  std::u32string u32_decoded = utf8_to_u32(decoded);

  // Boundary check: if substart exceeds decoded string length, skip this token
  if (substart > static_cast<int>(u32_decoded.size())) {
    skip_tokens_++;
    return "";
  }

  std::u32string u32_incremental = u32_decoded.substr(substart);
  std::string incremental = u32_to_utf8(u32_incremental);

  // Check if valid: non-empty and last character is valid CJK/ASCII letter/digit
  if (!incremental.empty() && is_valid_char(u32_incremental.back())) {
    // Valid: update last_response_ (decode last slide_len tokens)
    int resp_start =
        std::max(0, static_cast<int>(generated_ids_.size()) - kSlideLen);
    std::vector<Token> resp_window(generated_ids_.begin() + resp_start,
                                   generated_ids_.end());
    last_response_ = tokenizer_->decode(resp_window);
    skip_tokens_ = 0;
    return incremental;
  } else {
    // Invalid: skip, accumulate for next time
    skip_tokens_++;
    return "";
  }
}

void StreamingDecoder::reset() {
  generated_ids_.clear();
  last_response_.clear();
  skip_tokens_ = 0;
}

void StreamingDecoder::init(const std::vector<Token>& tokens) {
  generated_ids_ = tokens;
  // Initialize last_response_ by decoding the last kSlideLen prompt tokens
  int init_start = std::max(0, static_cast<int>(tokens.size()) - kSlideLen);
  std::vector<Token> init_window(tokens.begin() + init_start, tokens.end());
  last_response_ = tokenizer_->decode(init_window);
  skip_tokens_ = 0;
}

bool StreamingDecoder::is_valid_char(char32_t cp) {
  return
      // CJK Unified Ideographs
      (cp >= 0x4E00u && cp <= 0x9FFFu) ||
      (cp >= 0x3400u && cp <= 0x4DBFu) ||
      (cp >= 0x20000u && cp <= 0x2A6DFu) ||
      (cp >= 0x2A700u && cp <= 0x2B73Fu) ||
      (cp >= 0x2B740u && cp <= 0x2B81Fu) ||
      (cp >= 0x2B820u && cp <= 0x2CEAFu) ||
      // CJK Compatibility Ideographs
      (cp >= 0xF900u && cp <= 0xFAFFu) ||
      (cp >= 0x2F800u && cp <= 0x2FA1Fu) ||
      // CJK Symbols and Punctuation (U+3000-U+303F)
      (cp >= 0x3000u && cp <= 0x303Fu) ||
      // Halfwidth and Fullwidth Forms (contains fullwidth punctuation like ？，。)
      (cp >= 0xFF00u && cp <= 0xFFEFu) ||
      // Enclosed Digits (①②③④⑤⑥⑦⑧⑨⑩ etc.)
      (cp >= 0x2460u && cp <= 0x24FFu) ||
      // ASCII Letters
      (cp >= 0x0041u && cp <= 0x005Au) ||  // A-Z
      (cp >= 0x0061u && cp <= 0x007Au) ||  // a-z
      // ASCII Digits
      (cp >= 0x0030u && cp <= 0x0039u) ||  // 0-9
      // ASCII Punctuation & Mathematical Symbols
      (cp >= 0x0020u && cp <= 0x002Fu) ||  // Space ! " # $ % & ' ( ) * + , - . /
      (cp >= 0x003Au && cp <= 0x003Fu) ||  // : ; < = > ?
      (cp >= 0x005Bu && cp <= 0x005Eu) ||  // [ \ ] ^
      cp == 0x007Eu;                        // ~ (tilde)
}

}  // namespace houmo
