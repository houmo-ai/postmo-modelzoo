/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_text_frontend.h
 * Description:
 *   Text frontend module for CosyVoice3 TTS demo.
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

#ifndef HM_TEXT_FRONTEND_H_
#define HM_TEXT_FRONTEND_H_

#include <memory>
#include <string>
#include <vector>

namespace tokenizers {
class Tokenizer;
}

namespace houmo {

/**
 * @brief Text frontend module for CosyVoice3 TTS.
 *
 * Handles:
 * - Text tokenization (encode/decode)
 * - Text normalization (Chinese and English)
 * - Paragraph splitting by token length
 */
class HmTextFrontend {
 public:
  /**
   * @brief Constructor - loads tokenizer from JSON file.
   * @param tokenizer_path Path to tokenizer.json file.
   */
  explicit HmTextFrontend(const std::string& tokenizer_path);

  /**
   * @brief Destructor.
   */
  ~HmTextFrontend();

  /**
   * @brief Encode text into token IDs.
   * Reference: demo.py lines 1199-1203
   * @param text Input text string.
   * @return Vector of token IDs.
   */
  std::vector<int> Encode(const std::string& text);

  /**
   * @brief Decode token IDs back to text.
   * Reference: demo.py lines 1205-1211
   * @param tokens Vector of token IDs.
   * @return Decoded text string.
   */
  std::string Decode(const std::vector<int>& tokens);

  /**
   * @brief Normalize text and optionally split into segments.
   * Reference: demo.py lines 1377-1430
   * @param text Input text to normalize.
   * @param split Whether to split into segments (default true).
   * @return Vector of normalized text segments (or single segment if
   * split=false).
   */
  std::vector<std::string> TextNormalize(const std::string& text,
                                         bool split = true);

  /**
   * @brief Split text by token length constraints.
   * Reference: demo.py lines 377-462
   * @param text Input text to split.
   * @param max_tokens Maximum tokens per segment (default 80).
   * @param min_tokens Minimum tokens before starting new segment (default 60).
   * @return Vector of text segments.
   */
  std::vector<std::string> SplitParagraph(const std::string& text,
                                          int max_tokens = 80,
                                          int min_tokens = 60);

  /**
   * @brief Check if tokenizer is loaded successfully.
   * @return True if tokenizer is ready.
   */
  bool IsLoaded() const;

 private:
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  /**
   * @brief Normalize Chinese text.
   * Simplified implementation without wetext library.
   * @param text Chinese text to normalize.
   * @return Normalized text.
   */
  std::string NormalizeChinese(const std::string& text);

  /**
   * @brief Normalize English text.
   * Simplified implementation - number-to-words skipped for now.
   * @param text English text to normalize.
   * @return Normalized text.
   */
  std::string NormalizeEnglish(const std::string& text);

  // Static helper functions (reference: demo.py helper functions)

  /**
   * @brief Check if text contains Chinese characters.
   * Reference: demo.py lines 305-308
   */
  static bool ContainsChinese(const std::string& text);

  /**
   * @brief Replace superscript symbols with spoken equivalents.
   * Reference: demo.py lines 311-315
   */
  static std::string ReplaceCornerMark(const std::string& text);

  /**
   * @brief Remove decorative brackets.
   * Reference: demo.py lines 318-324
   */
  static std::string RemoveBracket(const std::string& text);

  /**
   * @brief Keep spaces only between ASCII characters.
   * Reference: demo.py lines 327-338
   */
  static std::string ReplaceBlank(const std::string& text);

  /**
   * @brief Check if string is only punctuation/symbols.
   * Reference: demo.py lines 341-345
   */
  static bool IsOnlyPunctuation(const std::string& text);

  /**
   * @brief Load file contents as string.
   * @param path File path.
   * @return File contents.
   */
  static std::string LoadBytesFromFile(const std::string& path);

  // Internal split paragraph helpers
  int CalcUttLength(const std::string& text, bool is_chinese);
  bool ShouldMerge(const std::string& text, bool is_chinese, int merge_len);
};

}  // namespace houmo

#endif  // HM_TEXT_FRONTEND_H_