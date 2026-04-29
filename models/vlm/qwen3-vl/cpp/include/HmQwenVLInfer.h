/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmQwenVLInfer.h
 * Description:
 *   Main inference class for Qwen3-VL model.
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

#ifndef __HMQWENVLINFER_H__
#define __HMQWENVLINFER_H__

#include <iomanip>
#include <map>
#include <regex>

#include "HmImageProcessor.h"
#include "HmQwenVLTokenizer.h"
#include "SamplingManager.h"
#include "tcim/tcim_runtime.h"
#include "utils.h"

/**
 * @brief Timer class for performance measurement
 */
class Timer {
 public:
  void start() { start_time_ = std::chrono::high_resolution_clock::now(); }
  void end() { end_time_ = std::chrono::high_resolution_clock::now(); }
  float elapsed_ms() const {
    return std::chrono::duration<float, std::milli>(end_time_ - start_time_)
        .count();
  }

 private:
  std::chrono::high_resolution_clock::time_point start_time_;
  std::chrono::high_resolution_clock::time_point end_time_;
};

/**
 * @brief Main class for Qwen3-VL model inference
 *
 * This class implements the complete vision-language inference pipeline:
 * 1. Image preprocessing and vision model inference
 * 2. Prefill phase for processing input tokens
 * 3. Decode phase for generating output tokens
 */
class HmQwenVLInfer {
 public:
  /**
   * @brief Constructor
   * @param visualModelPath Path to visual (ViT) model
   * @param prefillModelPath Path to prefill model
   * @param decodeModelPath Path to decode model
   * @param tokenizerJsonPath Path to tokenizer JSON
   * @param embeddingWeightPath Path to embedding weights
   * @param sampling_manager Sampling manager for generation
   */
  HmQwenVLInfer(const std::string &visualModelPath,
                const std::string &prefillModelPath,
                const std::string &decodeModelPath,
                const std::string &tokenizerJsonPath,
                const std::string &embeddingWeightPath,
                const SamplingManager &sampling_manager = SamplingManager());

  HmQwenVLInfer(const HmQwenVLInfer &it) = delete;
  HmQwenVLInfer &operator=(const HmQwenVLInfer &it) = delete;
  HmQwenVLInfer(HmQwenVLInfer &&it) noexcept = default;
  HmQwenVLInfer &operator=(HmQwenVLInfer &&it) noexcept = default;
  ~HmQwenVLInfer();

  /**
   * @brief Run chat inference with image and text
   * @param image_paths Vector of image paths (can be empty for text-only)
   * @param prompt User text prompt
   * @return Generated response text
   */
  std::string Chat(const std::vector<std::string> &image_paths,
                   const std::string &prompt);

  void SetKeepHistory(bool keep_history) { keep_history_ = keep_history; }
  bool GetKeepHistory() const { return keep_history_; }
  void SetMaxNewTokens(int max_new_tokens) {
    max_new_tokens_ = max_new_tokens > 0 ? max_new_tokens : 1;
  }
  int GetMaxNewTokens() const { return max_new_tokens_; }
  void ResetConversationState() {
    context_length = 0;
    past_seq_len_ = 0;
    rope_deltas_ = 0;
    generated_ids_.clear();
    skip_tokens_ = 0;
    last_response_.clear();
  }
  void SetEnablePerfReport(bool enable_perf_report) {
    enable_perf_report_ = enable_perf_report;
  }
  bool GetEnablePerfReport() const { return enable_perf_report_; }

  /**
   * @brief Get performance metrics
   * @return PerfInfos structure with timing data
   */
  PerfInfos GetPerfInfo() const { return perf_info_; }

  /**
   * @brief Print model input/output information for debugging
   * @param module Model module
   * @param modelName Model name for display
   */
  void DebugModelInfo(tcim::Module &module, const std::string &modelName);

 private:
  // Model paths
  std::string visual_model_path_;
  std::string prefill_model_path_;
  std::string decode_model_path_;

  // Configuration parameters
  int prefill_length_ = 0;
  int embedding_length_ = 0;
  int context_max_length_ = 0;
  int batch_ = 0;
  int argmax_dim_len_ = 0;
  int attn_idx_start_ = 0;
  int vision_input_nums_ = 0;

  // Model modules
  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> visual_module_;
  std::shared_ptr<tcim::Module> prefill_module_;
  std::shared_ptr<tcim::Module> decode_module_;

  // Components
  std::shared_ptr<HmQwenVLTokenizer> tokenizer_;
  std::shared_ptr<HmImageProcessor> image_processor_;
  SamplingManager sampling_manager_;

  // Tensor maps
  std::map<std::string, tcim::Tensor> visual_input_map_;
  std::map<std::string, tcim::Tensor> prefill_input_map_;
  std::map<std::string, tcim::Tensor> decode_input_map_;

  // State
  std::vector<std::string> dummy_names_;
  int32_t past_seq_len_ = 0;
  std::vector<int> generated_ids_;
  int rope_deltas_ = 0;
  int slide_len_ = 10;
  int skip_tokens_ = 0;
  std::string last_response_;
  PerfInfos perf_info_;

  // Model config
  ModelConfig config_;

 private:
  // Initialization
  int GetNBlocks();
  int GetAttnIdxStart();
  void InitVisualInputs();
  void InitPrefillInputs();
  void InitDecodeInputs();

  // Vision model
  void VisionSetInput(const std::vector<half_float::half> &visual_data);
  void VisionInfer();
  std::tuple<std::vector<half_float::half>, std::vector<half_float::half>,
             std::vector<half_float::half>, std::vector<half_float::half>>
  VisionGetOutputs();

  // Prefill model
  void PrefillSetInputDatas(
      const std::vector<half_float::half> &inputs_embeds,
      const std::vector<int32_t> &time_position_ids,
      const std::vector<int32_t> &height_position_ids,
      const std::vector<int32_t> &width_position_ids, int32_t valid_length,
      int32_t current_length,
      const std::vector<half_float::half> &deepstack_embed_0,
      const std::vector<half_float::half> &deepstack_embed_1,
      const std::vector<half_float::half> &deepstack_embed_2);
  void PrefillInfer();
  int PrefillGetOutputDatas();

  // Decode model
  void DecodeSetInputDatas(
      const std::vector<half_float::half> &inputs_embeds,
      const std::vector<int32_t> &time_position_ids,
      const std::vector<int32_t> &height_position_ids,
      const std::vector<int32_t> &width_position_ids, int32_t valid_length,
      const std::vector<half_float::half> &deepstack_embed_0,
      const std::vector<half_float::half> &deepstack_embed_1,
      const std::vector<half_float::half> &deepstack_embed_2);
  void DecodeInfer();
  int DecodeGetOutputDatas();
  int context_length = 0;
  bool keep_history_ = true;
  int max_new_tokens_ = 256;
  bool enable_perf_report_ = true;
  // Utility
  bool IsValidChar(char32_t cp);
};

#endif  // __HMQWENVLINFER_H__