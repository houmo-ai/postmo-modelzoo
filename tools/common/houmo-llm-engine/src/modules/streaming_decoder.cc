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

#include <utf8proc/utf8proc.h>

namespace houmo {

// ============================================================================
// UTF-8 helpers
// ============================================================================

bool StreamingDecoder::is_valid_utf8(const std::string& s) {
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(s.data());
  const uint8_t* end = ptr + s.size();
  while (ptr < end) {
    int32_t cp = 0;
    utf8proc_ssize_t len = utf8proc_iterate(ptr, end - ptr, &cp);
    if (len < 0) return false;
    ptr += len;
  }
  return true;
}

bool StreamingDecoder::is_valid_char(char32_t cp) {
  if (cp >= 0xD800u && cp <= 0xDFFFu) return false;
  if (cp == 0xFFFDu) return false;
  if (cp >= 0xFDD0u && cp <= 0xFDEFu) return false;
  return true;
}

char32_t StreamingDecoder::last_codepoint(const std::string& s) {
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(s.data());
  const uint8_t* end = ptr + s.size();
  int32_t cp = 0;
  while (ptr < end) {
    int32_t cur = 0;
    utf8proc_ssize_t len = utf8proc_iterate(ptr, end - ptr, &cur);
    if (len < 0) break;
    cp = cur;
    ptr += len;
  }
  return static_cast<char32_t>(cp);
}

// ============================================================================
// StreamingDecoder implementation
// ============================================================================

StreamingDecoder::StreamingDecoder(std::shared_ptr<HfTokenizer> tokenizer)
    : tokenizer_(tokenizer) {}

std::string StreamingDecoder::decode(Token token) {
  if (!tokenizer_) {
    return "";
  }

  generated_ids_.push_back(token);
  pending_ids_.push_back(token);

  std::string decoded = tokenizer_->decode(pending_ids_);

  if (!is_valid_utf8(decoded)) {
    return "";
  }

  char32_t last_cp = last_codepoint(decoded);
  if (!is_valid_char(last_cp)) {
    return "";
  }

  pending_ids_.clear();
  return decoded;
}

void StreamingDecoder::reset() {
  generated_ids_.clear();
  pending_ids_.clear();
}

void StreamingDecoder::init(const std::vector<Token>& tokens) {
  generated_ids_ = tokens;
  pending_ids_.clear();
}

}  // namespace houmo
