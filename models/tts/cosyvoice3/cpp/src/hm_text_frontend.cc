/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_text_frontend.cc
 * Description:
 *   Text frontend module implementation for CosyVoice3 TTS demo.
 *   Handles text normalization, tokenization, and paragraph splitting.
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

#include "hm_text_frontend.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <regex>
#include <sstream>

// tokenizers-cpp header
#include "tokenizers_cpp.h"

namespace houmo {

namespace {

std::string TrimAsciiWhitespace(const std::string& text) {
  size_t start = text.find_first_not_of(" \t\n\r\f\v");
  if (start == std::string::npos) {
    return "";
  }

  size_t end = text.find_last_not_of(" \t\n\r\f\v");
  return text.substr(start, end - start + 1);
}

void ReplaceAll(std::string* text, const std::string& from,
                const std::string& to) {
  if (text == nullptr || from.empty()) {
    return;
  }

  size_t pos = 0;
  while ((pos = text->find(from, pos)) != std::string::npos) {
    text->replace(pos, from.length(), to);
    pos += to.length();
  }
}

bool StartsWithAt(const std::string& text, size_t pos,
                  const std::string& token) {
  return pos + token.length() <= text.length() &&
         text.compare(pos, token.length(), token) == 0;
}

size_t Utf8CharLen(const std::string& text, size_t pos) {
  if (pos >= text.size()) {
    return 0;
  }

  unsigned char c = static_cast<unsigned char>(text[pos]);
  if (c < 0x80) {
    return 1;
  }
  if ((c & 0xE0) == 0xC0) {
    return 2;
  }
  if ((c & 0xF0) == 0xE0) {
    return 3;
  }
  if ((c & 0xF8) == 0xF0) {
    return 4;
  }
  return 1;
}

}  // namespace

// ============================================================================
// Constructor / Destructor
// ============================================================================

HmTextFrontend::HmTextFrontend(const std::string& tokenizer_path) {
  // Try loading tokenizer - support multiple formats
  // 1. First try tokenizer.json (FromBlobJSON)
  // 2. If not found, try vocab.json + merges.txt (FromBlobByteLevelBPE)

  std::string tokenizer_json_path = tokenizer_path;
  if (tokenizer_json_path.find("tokenizer.json") == std::string::npos) {
    // If path is a directory, look for tokenizer.json or CosyVoice-BlankEN
    // subdirectory
    if (tokenizer_json_path.back() != '/') {
      tokenizer_json_path += "/";
    }
    tokenizer_json_path += "CosyVoice-BlankEN/tokenizer.json";
  }

  // Try tokenizer.json first
  std::string json_bytes = LoadBytesFromFile(tokenizer_json_path);
  if (!json_bytes.empty()) {
    tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(json_bytes);
    if (tokenizer_) {
      std::cout << "[HmTextFrontend] Tokenizer loaded from JSON: "
                << tokenizer_json_path << std::endl;
      std::cout << "[HmTextFrontend] Vocabulary size: "
                << tokenizer_->GetVocabSize() << std::endl;
      return;
    }
  }
}

HmTextFrontend::~HmTextFrontend() = default;

// ============================================================================
// Tokenization (Encode / Decode)
// ============================================================================

std::vector<int> HmTextFrontend::Encode(const std::string& text) {
  if (!tokenizer_) {
    std::cerr << "[HmTextFrontend] Tokenizer not loaded, cannot encode"
              << std::endl;
    return {};
  }

  std::vector<int32_t> ids = tokenizer_->Encode(text);
  return std::vector<int>(ids.begin(), ids.end());
}

std::string HmTextFrontend::Decode(const std::vector<int>& tokens) {
  if (!tokenizer_) {
    std::cerr << "[HmTextFrontend] Tokenizer not loaded, cannot decode"
              << std::endl;
    return "";
  }

  // Convert int to int32_t for tokenizer API
  std::vector<int32_t> ids(tokens.begin(), tokens.end());

  return tokenizer_->Decode(ids);
}

bool HmTextFrontend::IsLoaded() const { return tokenizer_ != nullptr; }

// ============================================================================
// Text Normalization
// ============================================================================

std::vector<std::string> HmTextFrontend::TextNormalize(const std::string& text,
                                                       bool split) {
  // Python: skip text_frontend when SSML/special-token marker exists.
  if (text.find("<|") != std::string::npos &&
      text.find("|>") != std::string::npos) {
    return split ? std::vector<std::string>{text}
                 : std::vector<std::string>{text};
  }

  if (text.empty()) {
    return split ? std::vector<std::string>{""}
                 : std::vector<std::string>{text};
  }

  std::string normalized = TrimAsciiWhitespace(text);
  if (normalized.empty()) {
    return split ? std::vector<std::string>{""} : std::vector<std::string>{""};
  }

  // Check if contains Chinese
  bool is_chinese = ContainsChinese(normalized);

  if (is_chinese) {
    normalized = NormalizeChinese(normalized);
    if (split) {
      return SplitParagraph(normalized, 80, 60);
    }
    return {normalized};
  } else {
    normalized = NormalizeEnglish(normalized);
    if (split) {
      // For English, we use token count for length calculation
      std::vector<std::string> segments;
      // Simplified: use same split logic with token-based length
      segments = SplitParagraph(normalized, 80, 60);
      return segments;
    }
    return {normalized};
  }
}

// ============================================================================
// Chinese Text Normalization
// ============================================================================

std::string HmTextFrontend::NormalizeChinese(const std::string& text) {
  // Simplified implementation without wetext library

  std::string result = text;

  // Remove newline characters
  result.erase(std::remove(result.begin(), result.end(), '\n'), result.end());

  // Replace blank (keep spaces only between ASCII)
  result = ReplaceBlank(result);

  // Replace corner marks (superscript symbols)
  result = ReplaceCornerMark(result);

  // Replace "." with Chinese period "。".
  ReplaceAll(&result, ".", "\xe3\x80\x82");

  // Replace " - " with Chinese comma "，".
  ReplaceAll(&result, " - ", "\xef\xbc\x8c");

  // Remove brackets
  result = RemoveBracket(result);

  // Remove trailing punctuation (replace multiple punctuation at end with
  std::regex trailing_punct_pattern("[，,、]+$");
  result = std::regex_replace(result, trailing_punct_pattern,
                              "\xe3\x80\x82");  // "。"

  return result;
}

// ============================================================================
// English Text Normalization
// ============================================================================

std::string HmTextFrontend::NormalizeEnglish(const std::string& text) {
  // Simplified: skip number-to-words conversion (inflect library)
  return text;
}

// ============================================================================
// Paragraph Splitting
// ============================================================================

std::vector<std::string> HmTextFrontend::SplitParagraph(const std::string& text,
                                                        int max_tokens,
                                                        int min_tokens) {
  const int merge_len = 20;

  bool is_chinese = ContainsChinese(text);

  // Define punctuation for splitting
  std::vector<std::string> punc;
  if (is_chinese) {
    punc = {"\xe3\x80\x82",
            "\xef\xbc\x9f",
            "\xef\xbc\x81",  // "。", "？", "！"
            "\xef\xbc\x9b",
            "\xef\xbc\x9a",
            "\xe3\x80\x81",  // "；", "：", "、"
            ".",
            "?",
            "!",
            ";"};  // ASCII punctuation
  } else {
    punc = {".", "?", "!", ";", ":"};
  }

  // Add ending punctuation if not present
  std::string work_text = text;
  bool has_ending_punc = false;
  for (const auto& p : punc) {
    if (work_text.length() >= p.length() &&
        work_text.compare(work_text.length() - p.length(), p.length(), p) ==
            0) {
      has_ending_punc = true;
      break;
    }
  }

  if (!has_ending_punc && !work_text.empty()) {
    if (is_chinese) {
      work_text += "\xe3\x80\x82";  // "。"
    } else {
      work_text += ".";
    }
  }

  // Split by punctuation, following demo.py semantics.
  std::vector<std::string> utts;
  size_t st = 0;

  for (size_t i = 0; i < work_text.length();) {
    bool matched = false;
    for (const auto& p : punc) {
      if (StartsWithAt(work_text, i, p)) {
        // Extract segment before punctuation
        if (i > st) {
          utts.push_back(work_text.substr(st, i - st) + p);
        }

        // Handle quote after punctuation: Python checks '"' and '”'.
        size_t next_pos = i + p.length();
        if (next_pos < work_text.length()) {
          if (StartsWithAt(work_text, next_pos, "\"") ||
              StartsWithAt(work_text, next_pos, "\xe2\x80\x9d")) {
            if (!utts.empty()) {
              size_t quote_len = Utf8CharLen(work_text, next_pos);
              utts.back() += work_text.substr(next_pos, quote_len);
              st = next_pos + quote_len;
            } else {
              st = next_pos;
            }
          } else {
            st = next_pos;
          }
        } else {
          st = next_pos;
        }
        i += p.length();
        matched = true;
        break;
      }
    }

    if (!matched) {
      i += Utf8CharLen(work_text, i);
    }
  }

  // Merge segments according to token length constraints
  std::vector<std::string> final_utts;
  std::string cur_utt;

  for (const auto& utt : utts) {
    int cur_len = CalcUttLength(cur_utt, is_chinese);
    int combined_len = CalcUttLength(cur_utt + utt, is_chinese);
    int utt_len = CalcUttLength(utt, is_chinese);

    if (combined_len > max_tokens && cur_len > min_tokens) {
      final_utts.push_back(cur_utt);
      cur_utt = "";
    }
    cur_utt += utt;
  }

  // Handle remaining segment
  if (!cur_utt.empty()) {
    if (ShouldMerge(cur_utt, is_chinese, merge_len) && !final_utts.empty()) {
      final_utts.back() += cur_utt;
    } else {
      final_utts.push_back(cur_utt);
    }
  }

  // Filter out punctuation-only segments
  std::vector<std::string> filtered_utts;
  for (const auto& utt : final_utts) {
    if (!IsOnlyPunctuation(utt)) {
      filtered_utts.push_back(utt);
    }
  }

  return filtered_utts;
}

int HmTextFrontend::CalcUttLength(const std::string& text, bool is_chinese) {
  if (is_chinese) {
    // For Chinese, use character count
    // Need to count UTF-8 characters properly
    int count = 0;
    for (size_t i = 0; i < text.length();) {
      unsigned char c = static_cast<unsigned char>(text[i]);
      if (c < 128) {
        count++;
        i++;
      } else if ((c & 0xE0) == 0xC0) {
        count++;
        i += 2;
      } else if ((c & 0xF0) == 0xE0) {
        count++;
        i += 3;
      } else if ((c & 0xF8) == 0xF0) {
        count++;
        i += 4;
      } else {
        i++;
      }
    }
    return count;
  } else {
    // For English, use token count
    return static_cast<int>(Encode(text).size());
  }
}

bool HmTextFrontend::ShouldMerge(const std::string& text, bool is_chinese,
                                 int merge_len) {
  return CalcUttLength(text, is_chinese) < merge_len;
}

// ============================================================================
// Static Helper Functions
// ============================================================================

bool HmTextFrontend::ContainsChinese(const std::string& text) {
  // Check for Chinese characters (Unicode range: \u4e00-\u9fff)
  // In UTF-8, Chinese characters are 3 bytes starting with 0xE4-0xE9

  for (size_t i = 0; i < text.length();) {
    unsigned char c = static_cast<unsigned char>(text[i]);

    // Check for 3-byte UTF-8 characters in Chinese range
    if ((c & 0xF0) == 0xE0) {
      if (i + 2 < text.length()) {
        // Chinese characters: \u4e00-\u9fff
        // UTF-8 encoding:
        // \u4e00: E4 B8 80
        // \u9fff: E9 BF BF
        unsigned char b1 = static_cast<unsigned char>(text[i + 1]);
        unsigned char b2 = static_cast<unsigned char>(text[i + 2]);

        // Check if in range E4-E9 with appropriate second byte range
        if (c >= 0xE4 && c <= 0xE9) {
          if (c == 0xE4 && b1 >= 0xB8) return true;  // \u4e00-\u4fff
          if (c >= 0xE5 && c <= 0xE8) return true;   // \u5000-\u8fff
          if (c == 0xE9 && b1 <= 0xBF) return true;  // \u9000-\u9fff
        }
      }
      i += 3;
    } else if (c < 128) {
      i++;
    } else if ((c & 0xE0) == 0xC0) {
      i += 2;
    } else if ((c & 0xF8) == 0xF0) {
      i += 4;
    } else {
      i++;
    }
  }
  return false;
}

std::string HmTextFrontend::ReplaceCornerMark(const std::string& text) {
  std::string result = text;

  // Replace superscript 2 ("²") with Chinese word for "square"
  // UTF-8: ² = \xc2\xb2, Chinese "square" = \xe5\xb9\xb3\xe6\x96\xb9
  size_t pos = 0;
  while ((pos = result.find("\xc2\xb2", pos)) != std::string::npos) {
    result.replace(pos, 2, "\xe5\xb9\xb3\xe6\x96\xb9");  // Chinese "square"
    pos += 6;
  }

  // Replace superscript 3 ("³") with Chinese word for "cube"
  // UTF-8: ³ = \xc2\xb3, Chinese "cube" = \xe7\xab\x8b\xe6\x96\xb9
  pos = 0;
  while ((pos = result.find("\xc2\xb3", pos)) != std::string::npos) {
    result.replace(pos, 2, "\xe7\xab\x8b\xe6\x96\xb9");  // Chinese "cube"
    pos += 6;
  }

  return result;
}

std::string HmTextFrontend::RemoveBracket(const std::string& text) {
  std::string result = text;

  // Remove Chinese full-width brackets: （ ） 【 】
  // UTF-8 codes:
  // （ = \xef\xbc\x88
  // ） = \xef\xbc\x89
  // 【 = \xe3\x80\x90
  // 】 = \xe3\x80\x91

  std::vector<std::pair<std::string, std::string>> replacements = {
      {"\xef\xbc\x88", ""},              // Remove "（"
      {"\xef\xbc\x89", ""},              // Remove "）"
      {"\xe3\x80\x90", ""},              // Remove "【"
      {"\xe3\x80\x91", ""},              // Remove "】"
      {"`", ""},                         // Remove backtick
      {"\xe2\x80\x94\xe2\x80\x94", " "}  // Replace "——" with space
  };

  for (const auto& rep : replacements) {
    size_t pos = 0;
    while ((pos = result.find(rep.first, pos)) != std::string::npos) {
      result.replace(pos, rep.first.length(), rep.second);
      pos += rep.second.length();
    }
  }

  return result;
}

std::string HmTextFrontend::ReplaceBlank(const std::string& text) {
  // Keep spaces only when they separate adjacent ASCII segments

  std::string result;
  for (size_t i = 0; i < text.length(); ++i) {
    char c = text[i];

    if (c == ' ') {
      // Check if space is between two ASCII non-space characters
      bool prev_is_ascii = (i > 0) &&
                           (static_cast<unsigned char>(text[i - 1]) < 128) &&
                           (text[i - 1] != ' ');
      bool next_is_ascii = (i + 1 < text.length()) &&
                           (static_cast<unsigned char>(text[i + 1]) < 128) &&
                           (text[i + 1] != ' ');

      if (prev_is_ascii && next_is_ascii) {
        result += c;
      }
    } else {
      result += c;
    }
  }

  return result;
}

bool HmTextFrontend::IsOnlyPunctuation(const std::string& text) {
  // Check if string contains only punctuation/symbols

  // Common ASCII punctuation
  const std::string ascii_punct = ".,!?;:'\"()-[]{}<>@#$%^&*_+=|\\~/`";

  // Chinese punctuation (UTF-8)
  const std::vector<std::string> chinese_punct = {
      "\xe3\x80\x82",  // 。
      "\xef\xbc\x8c",  // ，
      "\xef\xbc\x9f",  // ？
      "\xef\xbc\x81",  // ！
      "\xef\xbc\x9b",  // ；
      "\xef\xbc\x9a",  // ：
      "\xe3\x80\x90",  // 【
      "\xe3\x80\x91",  // 】
      "\xef\xbc\x88",  // （
      "\xef\xbc\x89",  // ）
      "\xe3\x80\x8c",  // "
      "\xe3\x80\x8d",  // "
      "\xe2\x80\x9c",  // "
      "\xe2\x80\x9d",  // "
      "\xef\xbc\x8e",  // ．
  };

  if (text.empty()) {
    return true;
  }

  size_t i = 0;
  while (i < text.length()) {
    unsigned char c = static_cast<unsigned char>(text[i]);

    // Check ASCII punctuation
    if (c < 128) {
      if (ascii_punct.find(c) == std::string::npos && !std::isspace(c)) {
        return false;
      }
      i++;
      continue;
    }

    // Check for multi-byte punctuation
    bool found = false;
    for (const auto& p : chinese_punct) {
      if (i + p.length() <= text.length() &&
          text.compare(i, p.length(), p) == 0) {
        i += p.length();
        found = true;
        break;
      }
    }

    if (!found) {
      // Not a known punctuation character
      return false;
    }
  }

  return true;
}

std::string HmTextFrontend::LoadBytesFromFile(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "[HmTextFrontend] Cannot open file: " << path << std::endl;
    return "";
  }

  std::ostringstream ss;
  ss << file.rdbuf();
  return ss.str();
}

}  // namespace houmo