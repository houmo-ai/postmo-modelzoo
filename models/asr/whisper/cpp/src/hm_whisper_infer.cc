/*
 * Copyright (c) 2026 HOUMO AI
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * File: hm_whisper_infer.cpp
 * Description: Implementation of Whisper ASR inference class
 */

#include "hm_whisper_infer.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <unordered_set>

#include "SamplingManager.h"

namespace houmo {

// Language detection token IDs (from Whisper vocabulary)
static const std::vector<int> LANG_TO_ID = {
    50327, 50334, 50272, 50350, 50304, 50355, 50330, 50292, 50302, 50347,
    50309, 50315, 50270, 50283, 50297, 50285, 50261, 50281, 50259, 50262,
    50307, 50310, 50300, 50277, 50338, 50265, 50319, 50333, 50352, 50354,
    50279, 50276, 50291, 50339, 50286, 50312, 50275, 50311, 50274, 50266,
    50356, 50329, 50316, 50323, 50306, 50264, 50294, 50345, 50353, 50336,
    50293, 50301, 50349, 50295, 50308, 50296, 50314, 50320, 50282, 50343,
    50346, 50313, 50271, 50342, 50288, 50328, 50321, 50269, 50340, 50267,
    50284, 50263, 50344, 50332, 50322, 50298, 50305, 50324, 50326, 50317,
    50303, 50357, 50273, 50318, 50287, 50299, 50331, 50289, 50341, 50348,
    50268, 50351, 50280, 50290, 50337, 50278, 50335, 50325, 50260};

static int DetectLanguageId(TensorType* logits_data,
                            const std::vector<int>& lang_to_id,
                            int vocab_size) {
  std::vector<bool> non_lang_mask(vocab_size, true);
  for (int id : lang_to_id) {
    non_lang_mask[id] = false;
  }

  int lang_id = -1;
  float max_logit = -std::numeric_limits<float>::infinity();

  for (int i = 0; i < vocab_size; ++i) {
    if (non_lang_mask[i]) {
      logits_data[i] = static_cast<TensorType>(-65504.0f);
    }
    float current_val = static_cast<float>(logits_data[i]);
    if (current_val > max_logit) {
      max_logit = current_val;
      lang_id = i;
    }
  }
  return lang_id;
}

HmWhisperInfer::HmWhisperInfer(const std::string& encoder_path,
                               const std::string& decoder_path,
                               const std::string& prefill_path,
                               const std::string& tokenizer_path,
                               const SamplingParams& sampling_params)
    : encoder_path_(encoder_path),
      decoder_path_(decoder_path),
      prefill_path_(prefill_path),
      lang_to_id_(LANG_TO_ID),
      sampling_params_(sampling_params) {
  auto tokenizer_json = LoadBytesFromFile(tokenizer_path);
  tokenizer_ = tokenizers::Tokenizer::FromBlobJSON(tokenizer_json);

  weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);

  auto encoder_option = tcim::Module::Option(weight_manager_);
  auto decoder_option = tcim::Module::Option(weight_manager_);
  auto prefill_option = tcim::Module::Option(weight_manager_);

  encoder_module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_STATUS(encoder_module_->LoadModel(encoder_path_, encoder_option));

  prefill_module_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_STATUS(prefill_module_->LoadModel(prefill_path_, prefill_option));

  decoder_module_ = std::make_shared<tcim::Module>();
  decoder_module_->LoadModel(decoder_path_, decoder_option);

  InitMaps();

  for (int i = 0; i < 96; ++i) {
    const std::string layer_name = decoder_module_->GetInputName(i + 6);
    auto cache = decoder_module_->GetDevInput(layer_name);
    prefill_module_->SetDevInput(layer_name, cache);
  }
}

HmWhisperInfer::~HmWhisperInfer() {
  encoder_module_.reset();
  decoder_module_.reset();
  prefill_module_.reset();
}

void HmWhisperInfer::DebugModelInfo(tcim::Module& module,
                                    const std::string& model_name) {
  // DebugModelInfo disabled to reduce output
}

void HmWhisperInfer::InitMaps() {
  if (encoder_module_ == nullptr) return;
  encoder_input_map_.clear();
  for (int idx = 0; idx < encoder_module_->GetInputNum(); ++idx) {
    auto name = encoder_module_->GetInputName(idx);
    auto info = encoder_module_->GetInputInfo(name).AsContiguous();
    encoder_input_map_.insert({name, tcim::Tensor::CreateHostTensor(info)});
  }

  if (prefill_module_ == nullptr) return;
  prefill_input_map_.clear();
  for (int idx = 0; idx < 6; ++idx) {
    auto name = prefill_module_->GetInputName(idx);
    auto info = prefill_module_->GetInputInfo(name).AsContiguous();
    prefill_input_map_.insert({name, tcim::Tensor::CreateHostTensor(info)});
  }

  if (decoder_module_ == nullptr) return;
  decoder_input_map_.clear();
  for (int idx = 0; idx < decoder_module_->GetInputNum(); ++idx) {
    auto name = decoder_module_->GetInputName(idx);
    auto info = decoder_module_->GetInputInfo(name).AsContiguous();
    decoder_input_map_.insert({name, tcim::Tensor::CreateHostTensor(info)});
  }

  TensorType cache_data = static_cast<TensorType>(-65504.0f);
  for (int idx = 0; idx < decoder_module_->GetInputNum(); ++idx) {
    auto name = decoder_module_->GetInputName(idx);
    if (name.find("k_cache") != std::string::npos ||
        name.find("v_cache") != std::string::npos) {
      auto tensor = decoder_input_map_.at(name);
      TensorType* data = static_cast<TensorType*>(tensor.Buffer().Data());
      size_t num = tensor.MemSize() / sizeof(TensorType);
      std::fill(data, data + num, cache_data);
    }
  }

  // Pre-allocate reusable buffers to avoid repeated allocations per chunk
  const int vocab_size = tokenizer_->GetVocabSize();
  mask_attn_prefill_.resize(16 * 4 * 1024);
  encoder_attention_mask_.resize(1500);
  float_logits_.resize(vocab_size);
  float_decode_logits_.resize(vocab_size);
  default_decoder_ids_.resize(max_decode_length_);
  chat_history_ids_.reserve(max_decode_length_);
  loop_input_ids_.resize(1);
  loop_cache_pos_.resize(1);
  loop_past_len_.resize(1);
  loop_mask_attn_.resize(16 * 1024);
  decode_window_ids_.reserve(max_decode_length_);

  // Initialize mask_attn_prefill_ once
  std::fill(mask_attn_prefill_.begin(), mask_attn_prefill_.end(),
            static_cast<TensorType>(1.0f));
  for (int b = 0; b < 16; ++b) {
    for (int r = 0; r < 4; ++r) {
      for (int c = 4; c < 1024; ++c) {
        mask_attn_prefill_[b * 4 * 1024 + r * 1024 + c] =
            static_cast<TensorType>(-65504.0f);
      }
    }
  }

  // Initialize encoder_attention_mask_ once
  std::fill(encoder_attention_mask_.begin(), encoder_attention_mask_.end(),
            static_cast<TensorType>(0.0f));

  // Pre-allocate output tensors
  auto pref_logits_name = prefill_module_->GetOutputName(0);
  auto pref_logits_info =
      prefill_module_->GetOutputInfo(pref_logits_name).AsContiguous();
  pref_logits_tensor_ = tcim::Tensor::CreateHostTensor(pref_logits_info);

  auto dec_logits_name = decoder_module_->GetOutputName(0);
  auto dec_logits_info =
      decoder_module_->GetOutputInfo(dec_logits_name).AsContiguous();
  dec_logits_tensor_ = tcim::Tensor::CreateHostTensor(dec_logits_info);
}

bool HmWhisperInfer::IsValidChar(char32_t cp) {
  if (cp == 0xFFFDu || cp <= 0x001Fu) {
    return false;
  }
  return true;
}

std::vector<tcim::Tensor> HmWhisperInfer::RunEncoder(
    const std::vector<TensorType>& input_features, int n_mels, int n_frames) {
  EncoderSetInputs(input_features, n_mels, n_frames);

  auto t_start = std::chrono::high_resolution_clock::now();
  encoder_module_->Run();
  encoder_module_->Sync();
  auto t_end = std::chrono::high_resolution_clock::now();

  return EncoderGetOutputs();
}

void HmWhisperInfer::EncoderSetInputs(
    const std::vector<TensorType>& input_features, int n_mels, int n_frames) {
  std::string name = encoder_module_->GetInputName(0);
  auto tensor = encoder_input_map_[name];
  CHECK_TCIM_STATUS(
      tensor.Buffer().CopyFromHost(input_features.data(), tensor.MemSize()));
  encoder_module_->SetInput(name, tensor);
}

std::vector<tcim::Tensor> HmWhisperInfer::EncoderGetOutputs() {
  std::vector<tcim::Tensor> outputs;
  int output_num = encoder_module_->GetOutputNum();

  for (int idx = 0; idx < output_num; idx++) {
    std::string output_name = encoder_module_->GetOutputName(idx);
    int is_value = idx % 2;
    int layer_idx = idx / 2;
    int decoder_input_idx = is_value ? (78 + layer_idx) : (54 + layer_idx);

    std::string decoder_name = decoder_module_->GetInputName(decoder_input_idx);
    auto output_tensor = decoder_input_map_.at(decoder_name);

    encoder_module_->GetOutput(output_name).CastTo(output_tensor);
    outputs.push_back(output_tensor);
  }
  return outputs;
}

void HmWhisperInfer::RunDecoder(
    const std::vector<tcim::Tensor>& encoder_outputs) {
  std::vector<int> decode_input_ids = {
      tokenizer_->TokenToId("<|startoftranscript|>")};
  std::vector<int> cache_position = {0};
  std::vector<int> past_len = {0};
  std::vector<int> current_len = {1};
  std::vector<TensorType> mask_attn(16 * 1024, static_cast<TensorType>(1.0f));

  for (int i = 1; i < 1024; ++i) {
    for (int j = 0; j < 16; ++j) {
      mask_attn[j * 1024 + i] = static_cast<TensorType>(-65504.0f);
    }
  }

  std::vector<TensorType> encoder_attention_mask(1500,
                                                 static_cast<TensorType>(0.0f));

  for (int i = 0; i < decoder_module_->GetInputNum(); i++) {
    auto name = decoder_module_->GetInputName(i);
    auto tensor = decoder_input_map_.at(name);
    auto memSize = tensor.MemSize();

    if (name.find("input_ids") != std::string::npos) {
      CHECK_TCIM_STATUS(
          tensor.Buffer().CopyFromHost(decode_input_ids.data(), memSize));
    } else if (name.find("cache_position") != std::string::npos) {
      CHECK_TCIM_STATUS(
          tensor.Buffer().CopyFromHost(cache_position.data(), memSize));
    } else if (name.find("past_len") != std::string::npos) {
      CHECK_TCIM_STATUS(tensor.Buffer().CopyFromHost(past_len.data(), memSize));
    } else if (name.find("current_len") != std::string::npos) {
      CHECK_TCIM_STATUS(
          tensor.Buffer().CopyFromHost(current_len.data(), memSize));
    } else if (name.find("mask_attn") != std::string::npos) {
      CHECK_TCIM_STATUS(
          tensor.Buffer().CopyFromHost(mask_attn.data(), memSize));
    } else if (name.find("encoder_attention_mask") != std::string::npos) {
      CHECK_TCIM_STATUS(
          tensor.Buffer().CopyFromHost(encoder_attention_mask.data(), memSize));
    }

    decoder_module_->SetInput(name, tensor);
  }

  decoder_module_->Run();
  decoder_module_->Sync();
}

tcim::Tensor HmWhisperInfer::DecoderGetOutput() {
  std::string output_name = decoder_module_->GetOutputName(0);
  auto output_info = decoder_module_->GetOutputInfo(output_name).AsContiguous();
  tcim::Tensor output_tensor = tcim::Tensor::CreateHostTensor(output_info);
  decoder_module_->GetOutput(output_name).CastTo(output_tensor);
  return output_tensor;
}

std::pair<std::string, WhisperPerfInfo> HmWhisperInfer::Transcribe(
    const MelFeatures& mel_features, DecodeState* state,
    const std::string& language) {
  int slide_len = 10;
  int skip_tokens = 0;
  std::string last_response;
  auto t_start = std::chrono::high_resolution_clock::now();

  WhisperPerfInfo perf_info;
  perf_info.audio_duration = mel_features.n_frames * 0.02f;

  DecodeState local_state;
  if (state == nullptr) {
    state = &local_state;
  }

  auto encoder_outputs =
      RunEncoder(mel_features.data, mel_features.n_mels, mel_features.n_frames);

  RunDecoder(encoder_outputs);
  auto decoder_output = DecoderGetOutput();
  TensorType* logits_data =
      static_cast<TensorType*>(decoder_output.Buffer().Data());

  int vocab_size = tokenizer_->GetVocabSize();
  int end_of_text_id = tokenizer_->TokenToId("<|endoftext|>");

  int lang_id = tokenizer_->TokenToId("<|en|>");  // Default fallback to <|en|>
  if (language == "auto") {
    lang_id = DetectLanguageId(logits_data, lang_to_id_, vocab_size);
  } else {
    std::string lang_token = "<|" + language + "|>";
    int mapped_id = tokenizer_->TokenToId(lang_token);
    if (mapped_id != -1) {
      lang_id = mapped_id;
    } else {
      std::cerr << "Warning: Could not map language token: " << lang_token
                << " falling back to auto detect." << std::endl;
      lang_id = DetectLanguageId(logits_data, lang_to_id_, vocab_size);
    }
  }

  // Prefill phase
  std::vector<int> default_decoder_ids = {
      tokenizer_->TokenToId("<|startoftranscript|>"), lang_id,
      tokenizer_->TokenToId("<|transcribe|>"),
      tokenizer_->TokenToId("<|notimestamps|>")};
  std::vector<int> cache_position_prefill = {0, 1, 2, 3};
  std::vector<int> past_len = {0};
  std::vector<int> current_len = {4};
  std::vector<TensorType> mask_attn_prefill(16 * 4 * 1024,
                                            static_cast<TensorType>(1.0f));

  for (int b = 0; b < 16; ++b) {
    for (int r = 0; r < 4; ++r) {
      for (int c = 4; c < 1024; ++c) {
        mask_attn_prefill[b * 4 * 1024 + r * 1024 + c] =
            static_cast<TensorType>(-65504.0f);
      }
    }
  }

  std::vector<TensorType> encoder_attention_mask(1500,
                                                 static_cast<TensorType>(0.0f));

  auto set_prefill_input = [&](int idx, const void* data, size_t size) {
    auto name = prefill_module_->GetInputName(idx);
    auto tensor = prefill_input_map_.at(name);
    CHECK_TCIM_STATUS(tensor.Buffer().CopyFromHost(data, size));
    prefill_module_->SetInput(name, tensor);
  };

  set_prefill_input(0, default_decoder_ids.data(),
                    default_decoder_ids.size() * sizeof(int));
  set_prefill_input(1, cache_position_prefill.data(),
                    cache_position_prefill.size() * sizeof(int));
  set_prefill_input(2, past_len.data(), past_len.size() * sizeof(int));
  set_prefill_input(3, current_len.data(), current_len.size() * sizeof(int));
  set_prefill_input(4, mask_attn_prefill.data(),
                    mask_attn_prefill.size() * sizeof(TensorType));
  set_prefill_input(5, encoder_attention_mask.data(),
                    encoder_attention_mask.size() * sizeof(TensorType));

  auto t_prefill_start = std::chrono::high_resolution_clock::now();
  prefill_module_->Run();
  prefill_module_->Sync();
  auto t_prefill_end = std::chrono::high_resolution_clock::now();
  perf_info.ttft_time =
      std::chrono::duration<float, std::milli>(t_prefill_end - t_start).count();

  // Get prefill logits
  auto pref_logits_name = prefill_module_->GetOutputName(0);
  auto pref_logits_info =
      prefill_module_->GetOutputInfo(pref_logits_name).AsContiguous();
  tcim::Tensor pref_logits_tensor =
      tcim::Tensor::CreateHostTensor(pref_logits_info);
  prefill_module_->GetOutput(pref_logits_name).CastTo(pref_logits_tensor);
  TensorType* p_logits =
      static_cast<TensorType*>(pref_logits_tensor.Buffer().Data());

  SamplingManager sampling_manager(
      sampling_params_.temperature, sampling_params_.top_k,
      sampling_params_.top_p, sampling_params_.repetition_penalty,
      sampling_params_.min_tokens_to_keep);

  // Convert TensorType logits to float for SamplingManager
  std::vector<float> float_logits(vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    float_logits[i] = static_cast<float>(p_logits[3 * vocab_size + i]);
  }

  std::vector<int> cur_slide_win(
      default_decoder_ids.end() -
          std::min(slide_len, static_cast<int>(default_decoder_ids.size())),
      default_decoder_ids.end());
  if (last_response.empty()) {
    last_response = tokenizer_->Decode(cur_slide_win);
  }

  int next_token = sampling_manager.sample(float_logits.data(), vocab_size,
                                           default_decoder_ids);

  std::vector<int> chat_history_ids;
  default_decoder_ids.push_back(next_token);
  chat_history_ids.push_back(next_token);

  int window_size = slide_len + skip_tokens + 1;
  auto window_start = default_decoder_ids.size() >= window_size
                          ? default_decoder_ids.end() - window_size
                          : default_decoder_ids.begin();
  std::vector<int> decode_window_ids(window_start, default_decoder_ids.end());

  std::string tmp_response = tokenizer_->Decode(decode_window_ids);
  int substart = Utf8Len(last_response);
  std::u32string udecode_response = Utf8ToU32(tmp_response).substr(substart);
  std::string decode_response = U32ToUtf8(udecode_response);

  std::string all_response = "";
  if (decode_response != "" && IsValidChar(udecode_response.back()) &&
      next_token != end_of_text_id) {
    std::cout << decode_response << std::flush;
    all_response += decode_response;
    std::vector<int> cur_slide_win_new(
        default_decoder_ids.end() -
            std::min(slide_len, static_cast<int>(default_decoder_ids.size())),
        default_decoder_ids.end());
    last_response = tokenizer_->Decode(cur_slide_win_new);
    skip_tokens = 0;
  } else {
    skip_tokens += 1;
  }

  if (next_token == end_of_text_id) {
    decode_response = "";
  }

  // Transfer caches to decoder
  for (int i = 0; i < 48; ++i) {
    auto name = prefill_module_->GetOutputName(i + 1);
    auto cache = prefill_module_->GetDevOutput(name);
    decoder_module_->SetDevInput(decoder_module_->GetInputName(i + 6), cache);
    decoder_module_->SetDevOutput(decoder_module_->GetOutputName(i + 1), cache);
  }

  int cnt = 3;
  auto t_decode_start = std::chrono::high_resolution_clock::now();

  std::vector<int> loop_input_ids(1);
  std::vector<int> loop_cache_pos(1);
  std::vector<int> loop_past_len(1);
  std::vector<int> loop_curr_len = {1};
  std::vector<TensorType> loop_mask_attn(16 * 1024,
                                         static_cast<TensorType>(1.0f));

  auto set_decoder_input_by_name = [&](const std::string& name,
                                       const void* data, size_t size) {
    auto tensor = decoder_input_map_.at(name);
    CHECK_TCIM_STATUS(tensor.Buffer().CopyFromHost(data, size));
    decoder_module_->SetInput(name, tensor);
  };

  // Get input names for name-based matching
  std::string input_ids_name = decoder_module_->GetInputName(0);
  std::string cache_pos_name = decoder_module_->GetInputName(1);
  std::string past_len_name = decoder_module_->GetInputName(2);
  std::string curr_len_name = decoder_module_->GetInputName(3);
  std::string mask_attn_name = decoder_module_->GetInputName(4);
  std::string enc_attn_mask_name = decoder_module_->GetInputName(5);

  auto dec_logits_name = decoder_module_->GetOutputName(0);
  auto dec_logits_info =
      decoder_module_->GetOutputInfo(dec_logits_name).AsContiguous();
  tcim::Tensor dec_logits_tensor =
      tcim::Tensor::CreateHostTensor(dec_logits_info);

  while (default_decoder_ids.size() < 448 && next_token != end_of_text_id) {
    cnt++;
    loop_input_ids[0] = next_token;
    loop_cache_pos[0] = cnt;
    loop_past_len[0] = cnt;

    for (int b = 0; b < 16; ++b) {
      for (int c = 0; c < 1024; ++c) {
        loop_mask_attn[b * 1024 + c] = (c < cnt + 1)
                                           ? static_cast<TensorType>(1.0f)
                                           : static_cast<TensorType>(-65504.0f);
      }
    }

    set_decoder_input_by_name(input_ids_name, loop_input_ids.data(),
                              sizeof(int));
    set_decoder_input_by_name(cache_pos_name, loop_cache_pos.data(),
                              sizeof(int));
    set_decoder_input_by_name(past_len_name, loop_past_len.data(), sizeof(int));
    set_decoder_input_by_name(curr_len_name, loop_curr_len.data(), sizeof(int));
    set_decoder_input_by_name(mask_attn_name, loop_mask_attn.data(),
                              loop_mask_attn.size() * sizeof(TensorType));
    set_decoder_input_by_name(
        enc_attn_mask_name, encoder_attention_mask.data(),
        encoder_attention_mask.size() * sizeof(TensorType));

    decoder_module_->Run();
    decoder_module_->Sync();

    decoder_module_->GetOutput(dec_logits_name).CastTo(dec_logits_tensor);
    TensorType* d_logits =
        static_cast<TensorType*>(dec_logits_tensor.Buffer().Data());

    // Convert TensorType logits to float for SamplingManager
    std::vector<float> float_decode_logits(vocab_size);
    for (int i = 0; i < vocab_size; ++i) {
      float_decode_logits[i] = static_cast<float>(d_logits[i]);
    }
    next_token = sampling_manager.sample(float_decode_logits.data(), vocab_size,
                                         default_decoder_ids);

    default_decoder_ids.push_back(next_token);
    chat_history_ids.push_back(next_token);
    int substart = Utf8Len(last_response);

    int window_size = slide_len + skip_tokens + 1;
    auto window_start = default_decoder_ids.size() >= window_size
                            ? default_decoder_ids.end() - window_size
                            : default_decoder_ids.begin();
    std::vector<int> decode_window_ids(window_start, default_decoder_ids.end());

    std::string tmp_response = tokenizer_->Decode(decode_window_ids);
    std::u32string udecode_response = Utf8ToU32(tmp_response).substr(substart);
    std::string decode_response_str = U32ToUtf8(udecode_response);

    if (decode_response_str != "" && IsValidChar(udecode_response.back()) &&
        next_token != end_of_text_id) {
      std::cout << decode_response_str << std::flush;
      all_response += decode_response_str;
      std::vector<int> cur_slide_win(
          default_decoder_ids.end() -
              std::min(slide_len, static_cast<int>(default_decoder_ids.size())),
          default_decoder_ids.end());
      last_response = tokenizer_->Decode(cur_slide_win);
      skip_tokens = 0;
    } else {
      skip_tokens += 1;
    }
  }

  auto t_decode_end = std::chrono::high_resolution_clock::now();
  perf_info.decode_time =
      std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start)
          .count();
  perf_info.output_tokens = default_decoder_ids.size() -
                            4;  // prefill gives 4, and next tokens are appended

  state->last_response = last_response;
  state->skip_tokens = skip_tokens;

  return {all_response, perf_info};
}

}  // namespace houmo