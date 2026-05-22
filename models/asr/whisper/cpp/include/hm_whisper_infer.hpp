/*
 * Copyright (c) 2026 HOUMO AI
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * File: hm_whisper_infer.hpp
 * Description: Whisper ASR inference class using TCIM runtime
 */

#ifndef HM_WHISPER_INFER_HPP_
#define HM_WHISPER_INFER_HPP_

#include <chrono>
#include <codecvt>
#include <iomanip>
#include <locale>
#include <map>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

#include "hm_whisper_audio.hpp"
#include "tcim/tcim_runtime.h"
#include "tokenizers_cpp.h"

namespace houmo {

// Helper functions

/**
 * @brief Load file contents as bytes
 */
inline std::string LoadBytesFromFile(const std::string& path) {
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    std::cerr << "Cannot open " << path << "\n";
    throw std::runtime_error("Cannot open file: " + path);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

/**
 * @brief UTF-8 string length
 */
inline std::size_t Utf8Len(std::string_view u8) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.from_bytes(u8.data(), u8.data() + u8.size()).size();
}

/**
 * @brief UTF-8 to UTF-32 conversion
 */
inline std::u32string Utf8ToU32(const std::string& u8) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.from_bytes(u8);
}

/**
 * @brief UTF-32 to UTF-8 conversion
 */
inline std::string U32ToUtf8(const std::u32string& u32) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.to_bytes(u32);
}

/**
 * @brief Check TCIM status
 */
inline void CheckTcimStatus(const tcim::Status& status,
                            const char* file = __FILE__, int line = __LINE__) {
  if (status != tcim::Status::OK) {
    std::ostringstream err_msg;
    err_msg << "TCIM error at " << file << ":" << line
            << ", status: " << static_cast<int>(status);
    throw std::runtime_error(err_msg.str());
  }
}

#define CHECK_TCIM_STATUS(status) CheckTcimStatus(status, __FILE__, __LINE__)

/**
 * @brief Sampling parameters for text generation
 */
struct SamplingParams {
  float temperature = 1.0f;         ///< Temperature for scaling (>0)
  int top_k = -1;                   ///< Top-K threshold (-1 means disabled)
  float top_p = 1.0f;               ///< Top-P threshold (1.0 means disabled)
  float repetition_penalty = 1.1f;  ///< Repetition penalty (1.0 means disabled)
  int min_tokens_to_keep = 1;       ///< Minimum tokens to keep in top-p
};

/**
 * @brief Performance metrics for ASR inference
 */
struct WhisperPerfInfo {
  int prefill_tokens = 0;       ///< Prefill token count
  int output_tokens = 0;        ///< Total output tokens
  float ttft_time = 0.0f;       ///< Time to first token (ms)
  float prefill_time = 0.0f;    ///< Prefill phase time (ms)
  float decode_time = 0.0f;     ///< Decode phase time (ms)
  float encoder_time = 0.0f;    ///< Encoder time (ms)
  float audio_duration = 0.0f;  ///< Audio chunk duration (s)
};

/**
 * @brief Decoding state for cross-chunk continuity
 */
struct DecodeState {
  std::string last_response;  ///< Last decoded text
  int skip_tokens = 0;        ///< Tokens to skip for incremental decode
};

/**
 * @brief Whisper model inference class
 *
 * Handles:
 * - Model loading (encoder, decoder, prefill)
 * - Language detection
 * - Autoregressive decoding
 * - Token decoding with tokenizer
 */
class HmWhisperInfer {
 public:
  /**
   * @brief Constructor
   * @param encoder_path Path to encoder model (.hmm)
   * @param decoder_path Path to decoder model (.hmm)
   * @param prefill_path Path to prefill model (.hmm)
   * @param tokenizer_path Path to tokenizer JSON
   * @param sampling_params Sampling parameters (optional, uses defaults if not
   * provided)
   */
  HmWhisperInfer(const std::string& encoder_path,
                 const std::string& decoder_path,
                 const std::string& prefill_path,
                 const std::string& tokenizer_path,
                 const SamplingParams& sampling_params = SamplingParams());

  // Non-copyable
  HmWhisperInfer(const HmWhisperInfer&) = delete;
  HmWhisperInfer& operator=(const HmWhisperInfer&) = delete;

  // Moveable
  HmWhisperInfer(HmWhisperInfer&&) noexcept = default;
  HmWhisperInfer& operator=(HmWhisperInfer&&) noexcept = default;

  ~HmWhisperInfer();

  /**
   * @brief Get encoder module (for debugging)
   */
  std::shared_ptr<tcim::Module> GetEncoderModule() { return encoder_module_; }

  /**
   * @brief Get decoder module (for debugging)
   */
  std::shared_ptr<tcim::Module> GetDecoderModule() { return decoder_module_; }

  /**
   * @brief Get prefill module (for debugging)
   */
  std::shared_ptr<tcim::Module> GetPrefillModule() { return prefill_module_; }

  /**
   * @brief Print model I/O info (debug)
   */
  void DebugModelInfo(tcim::Module& module, const std::string& model_name);

  /**
   * @brief Run ASR transcription on mel features
   * @param mel_features Mel spectrogram
   * @param state Decoding state (optional, for chunk continuity)
   * @param language Forced language code (e.g. "zh", "en", "ko") or "auto"
   * @return Transcription and performance metrics
   */
  std::pair<std::string, WhisperPerfInfo> Transcribe(
      const MelFeatures& mel_features, DecodeState* state = nullptr,
      const std::string& language = "auto");

  /**
   * @brief Check if character is valid for display
   * @param cp Unicode code point
   * @return true if valid (CJK or ASCII letters)
   */
  static bool IsValidChar(char32_t cp);

 private:
  // Model paths
  std::string encoder_path_;
  std::string decoder_path_;
  std::string prefill_path_;

  // Tokenizer
  std::unique_ptr<tokenizers::Tokenizer> tokenizer_;

  // Configuration
  int n_blocks_ = 24;            ///< Transformer block count
  int max_decode_length_ = 448;  ///< Max decode tokens
  int eos_token_id_ = 50257;     ///< End-of-sequence token

  // Language IDs
  std::vector<int> lang_to_id_;

  // Sampling parameters
  SamplingParams sampling_params_;

  // Modules
  tcim::Module::WeightManager weight_manager_;
  std::shared_ptr<tcim::Module> encoder_module_;
  std::shared_ptr<tcim::Module> decoder_module_;
  std::shared_ptr<tcim::Module> prefill_module_;

  // Tensor maps
  std::map<std::string, tcim::Tensor> encoder_input_map_;
  std::map<std::string, tcim::Tensor> decoder_input_map_;
  std::map<std::string, tcim::Tensor> prefill_input_map_;

  // Pre-allocated buffers to avoid repeated allocations per chunk
  std::vector<TensorType> mask_attn_prefill_;       // 16 * 4 * 1024
  std::vector<TensorType> encoder_attention_mask_;  // 1500
  std::vector<float> float_logits_;                 // vocab_size (51865)
  std::vector<float> float_decode_logits_;          // vocab_size (51865)
  std::vector<int> default_decoder_ids_;            // max_decode_length_ (448)
  std::vector<int> chat_history_ids_;               // max_decode_length_ (448)
  std::vector<int> loop_input_ids_;                 // 1
  std::vector<int> loop_cache_pos_;                 // 1
  std::vector<int> loop_past_len_;                  // 1
  std::vector<TensorType> loop_mask_attn_;          // 16 * 1024
  std::vector<int> decode_window_ids_;              // slide_len + skip + 1

  // Pre-allocated tensors for outputs
  tcim::Tensor pref_logits_tensor_;
  tcim::Tensor dec_logits_tensor_;

  // Internal methods
  std::vector<tcim::Tensor> RunEncoder(
      const std::vector<TensorType>& input_features, int n_mels, int n_frames);
  void RunDecoder(const std::vector<tcim::Tensor>& encoder_outputs);
  tcim::Tensor DecoderGetOutput();
  void InitMaps();

  void EncoderSetInputs(const std::vector<TensorType>& input_features,
                        int n_mels, int n_frames);
  std::vector<tcim::Tensor> EncoderGetOutputs();
};

// Backward compatible aliases
using tensor_type = TensorType;
inline std::size_t utf8_len(std::string_view u8) { return Utf8Len(u8); }
inline std::u32string utf8_to_u32(const std::string& u8) {
  return Utf8ToU32(u8);
}
inline std::string u32_to_utf8(const std::u32string& u32) {
  return U32ToUtf8(u32);
}
inline void CheckTcimRetStatus(const tcim::Status& s, const char* f = __FILE__,
                               int l = __LINE__) {
  CheckTcimStatus(s, f, l);
}

}  // namespace houmo

#endif  // HM_WHISPER_INFER_HPP_