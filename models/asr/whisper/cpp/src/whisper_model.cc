/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: whisper_model.cc
 * Description:
 *   Whisper ASR model implementation
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

#include "whisper_model.h"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>

#include "base/tcim_utils.h"
#include "core/model_factory.h"
#include "modules/streaming_decoder.h"

namespace fs = std::filesystem;

namespace houmo {

namespace {

void save_bin(const std::string& path, const void* data, size_t size) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(data), size);
}

void save_tensor_bin(const std::string& path, const tcim::Tensor& tensor) {
  std::ofstream f(path, std::ios::binary);
  f.write(static_cast<const char*>(tensor.Buffer().Data()), tensor.MemSize());
}

}  // namespace

// ============================================================================
// Language Detection Token IDs (from Whisper vocabulary)
// These are the token IDs for language tokens like <|zh|>, <|en|>, etc.
// Used for language detection by masking non-language tokens.
// ============================================================================
static const std::vector<Token> LANG_TO_ID = {
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

// ============================================================================
// WhisperModel Implementation
// ============================================================================

WhisperModel::WhisperModel(const ModelConfig& config)
    : ASRModel(config), lang_to_id_(LANG_TO_ID) {
  load();
}

WhisperModel::~WhisperModel() {
  encoder_module_.reset();
  prefill_module_.reset();
  decode_module_.reset();
}

void WhisperModel::load() {
  // Step 1 - Initialize device manager
  dev_manager_ = std::make_unique<tcim::DevManager>(
      tcim::DevManager::Create(config_.devices));
  weight_manager_ = std::make_unique<tcim::Module::WeightManager>(
      tcim::Module::WeightManager::CreateWeightManager(*dev_manager_));

  // Step 2 - Load encode model
  {
    const std::string& encoder_path = config_.extra_params.at("encoder_path");

    auto encoder_option = tcim::Module::Option(*weight_manager_);
    encoder_option.EnableIOLazyMode(true);
    encoder_option.EnableHostLazyLoading(config_.lazy_mode);

    encoder_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        encoder_module_->LoadModel(encoder_path, encoder_option));

    // Get n_mels from encode input shape [1, n_mels, 3000]
    auto input0_shape =
        encoder_module_->GetInputInfo(encoder_module_->GetInputName(0)).Shape();
    if (input0_shape.size() >= 2) {
      n_mels_ = input0_shape[1];
      n_frames_ = input0_shape[2];
    }
  }

  // Step 3 - Load prefill model
  {
    const std::string& prefill_path = config_.prefill_path;

    auto prefill_option = tcim::Module::Option(*weight_manager_);
    prefill_option.EnableIOLazyMode(true);
    prefill_option.EnableHostLazyLoading(config_.lazy_mode);

    prefill_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        prefill_module_->LoadModel(prefill_path, prefill_option));
  }

  // Step 4 - Load decode model
  {
    const std::string& decode_path = config_.decode_path;

    auto decode_option = tcim::Module::Option(*weight_manager_);
    decode_option.EnableIOLazyMode(true);
    decode_option.EnableHostLazyLoading(config_.lazy_mode);

    decode_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        decode_module_->LoadModel(decode_path, decode_option));
    std::cout << "Decode model loaded: " << decode_path << std::endl;
  }

  // Step 5 - Share KV Cache between prefill and decode
  {
    // Find the base index for KV cache inputs
    int cache_count = 0;
    base_idx_ = -1;
    for (int idx = 0; idx < prefill_module_->GetInputNum(); ++idx) {
      const auto input_name = prefill_module_->GetInputName(idx);
      if (input_name.find("k_cache") != std::string::npos ||
          input_name.find("v_cache") != std::string::npos) {
        ++cache_count;
        if (base_idx_ < 0 && input_name.find("k_cache") == 0) {
          base_idx_ = idx;
        }
      }
    }
    if (base_idx_ < 0) {
      base_idx_ = 0;
    }
    num_decode_layers_ = cache_count / 2;

    // Share KV cache between prefill and decode modules
    const int cache_input_end =
        std::min(base_idx_ + 2 * num_decode_layers_,
                 static_cast<int>(std::min(prefill_module_->GetInputNum(),
                                           decode_module_->GetInputNum())));
    for (int i = base_idx_; i < cache_input_end; ++i) {
      const std::string layer_name = decode_module_->GetInputName(i);
      auto cache = decode_module_->GetDevInput(layer_name);
      CHECK_TCIM_RET_STATUS(prefill_module_->SetDevInput(layer_name, cache));
    }
  }

  // Step 6 - Parse model config
  {
    // Get num_heads and cache_max_len from prefill mask_attn input
    // mask_attn shape: [batch, num_heads, prompt_len, cache_max_len]
    if (prefill_module_->GetInputNum() > 4) {
      const auto mask_shape =
          prefill_module_->GetInputInfo(prefill_module_->GetInputName(4))
              .Shape();
      if (mask_shape.size() >= 4) {
        num_heads_ = mask_shape[1];
        cache_max_len_ = mask_shape[3];
      }
    }

    // Get encoder_seq_len from decoder encoder_attention_mask input
    if (decode_module_->GetInputNum() > 5) {
      const auto encoder_attn_shape =
          decode_module_->GetInputInfo(decode_module_->GetInputName(5)).Shape();
      if (!encoder_attn_shape.empty()) {
        encoder_seq_len_ = encoder_attn_shape[encoder_attn_shape.size() - 1];
      }
    }
  }

  // Step 7 - Initialize input tensors
  {
    // Initialize encode input map
    encoder_input_map_.clear();
    for (int idx = 0; idx < encoder_module_->GetInputNum(); ++idx) {
      auto name = encoder_module_->GetInputName(idx);
      auto info = encoder_module_->GetInputInfo(name).AsContiguous();
      encoder_input_map_[name] = tcim::Tensor::CreateHostTensor(info);
    }

    // Initialize prefill input map (only non-cache inputs)
    prefill_input_map_.clear();
    for (int idx = 0; idx < base_idx_; ++idx) {
      auto name = prefill_module_->GetInputName(idx);
      auto info = prefill_module_->GetInputInfo(name).AsContiguous();
      prefill_input_map_[name] = tcim::Tensor::CreateHostTensor(info);
    }

    // Initialize decode input map
    decode_input_map_.clear();
    for (int idx = 0; idx < decode_module_->GetInputNum(); ++idx) {
      auto name = decode_module_->GetInputName(idx);
      auto info = decode_module_->GetInputInfo(name).AsContiguous();
      decode_input_map_[name] = tcim::Tensor::CreateHostTensor(info);
    }

    // Initialize KV cache with -65504.0f (float16 min value)
    float16 cache_init_val = static_cast<float16>(-65504.0f);
    for (int idx = 0; idx < decode_module_->GetInputNum(); ++idx) {
      auto name = decode_module_->GetInputName(idx);
      if (name.find("k_cache") != std::string::npos ||
          name.find("v_cache") != std::string::npos) {
        auto tensor = decode_input_map_.at(name);
        float16* data = static_cast<float16*>(tensor.Buffer().Data());
        size_t num = tensor.MemSize() / sizeof(float16);
        std::fill(data, data + num, cache_init_val);
      }
    }

    std::cout << "Input tensors initialized" << std::endl;
  }

  // Step 8 - Load tokenizer
  {
    const std::string& tokenizer_path = config_.tokenizer_path;
    if (fs::exists(tokenizer_path)) {
      try {
        tokenizer_ = std::make_shared<HfTokenizer>(tokenizer_path);
        std::cout << "Tokenizer loaded from: " << tokenizer_path << std::endl;
      } catch (const Exception& e) {
        std::cerr << "Warning: Failed to load tokenizer from " << tokenizer_path
                  << ": " << e.what() << std::endl;
      }
    } else {
      std::cerr << "Warning: Tokenizer path does not exist: " << tokenizer_path
                << std::endl;
    }
  }

  // Step 9 - Initialize token IDs
  {
    if (!tokenizer_) {
      std::cerr << "Warning: Tokenizer not loaded, cannot init token IDs"
                << std::endl;
      return;
    }

    // Initialize special token IDs using token_to_id (direct lookup)
    sot_token_id_ = tokenizer_->token_to_id("<|startoftranscript|>");
    transcribe_token_id_ = tokenizer_->token_to_id("<|transcribe|>");
    notimestamps_token_id_ = tokenizer_->token_to_id("<|notimestamps|>");
    eos_token_id_ = tokenizer_->token_to_id("<|endoftext|>");
    // Build language token mapping
    // Format: "zh" -> <|zh|> token ID
    const std::vector<std::string> languages = {
        "en", "zh", "de", "es", "ru", "ko", "fr", "ja", "pt", "tr",  "pl", "ca",
        "nl", "ar", "sv", "it", "id", "hi", "fi", "vi", "he", "uk",  "el", "ms",
        "cs", "ro", "da", "hu", "ta", "no", "th", "ur", "hr", "bg",  "lt", "la",
        "mi", "ml", "cy", "sk", "te", "fa", "lv", "bn", "sr", "az",  "sl", "kn",
        "et", "mk", "br", "eu", "is", "hy", "ne", "mn", "bs", "kk",  "sq", "sw",
        "gl", "mr", "pa", "si", "km", "sn", "yo", "so", "af", "oc",  "ka", "be",
        "tg", "sd", "gu", "am", "yi", "lo", "uz", "fo", "ht", "ps",  "tk", "nn",
        "mt", "sa", "lb", "my", "bo", "tl", "mg", "as", "tt", "haw", "ln", "ha",
        "ba", "jw", "su", "yue"};

    for (const auto& lang : languages) {
      std::string lang_token = "<|" + lang + "|>";
      int id = tokenizer_->token_to_id(lang_token);
      if (id >= 0) {
        lang_token_map_[lang] = static_cast<Token>(id);
      }
    }

    default_lang_token_id_ = tokenizer_->token_to_id("<|zh|>");
  }
}

std::unique_ptr<Context> WhisperModel::create_context(int n_ctx) {
  if (n_ctx <= 0) {
    n_ctx = cache_max_len_;
  }
  return std::make_unique<WhisperContext>(this, n_ctx);
}

Token WhisperModel::lang_token_id(const std::string& language) const {
  auto it = lang_token_map_.find(language);
  if (it != lang_token_map_.end()) {
    return it->second;
  }
  return default_lang_token_id_;
}

// ============================================================================
// WhisperContext Implementation
// ============================================================================

WhisperContext::WhisperContext(ASRModel* model, int n_ctx)
    : ASRContext(model, n_ctx),
      audio_processor_(std::make_shared<AudioProcessor>(
          AudioProcessorConfig{.sample_rate = 16000,
                               .n_mels = model->n_mels(),
                               .chunk_seconds = 30,
                               .encoder_window_seconds = 30})) {}

MelFeatures WhisperContext::LoadAudio(const std::string& audio_path) {
  auto audio = audio_processor_->LoadAudio(audio_path);
  return audio_processor_->ExtractFeatures(audio);
}

void WhisperContext::set_audio_processor(int sample_rate, int chunk_seconds,
                                         int encoder_window_seconds) {
  auto config = audio_processor_->config();
  config.sample_rate = sample_rate;
  config.chunk_seconds = chunk_seconds;
  config.encoder_window_seconds = encoder_window_seconds;
  audio_processor_ = std::make_shared<AudioProcessor>(config);
}

int WhisperContext::sample_rate() const {
  return audio_processor_->sample_rate();
}

int WhisperContext::chunk_seconds() const {
  return audio_processor_->config().chunk_seconds;
}

int WhisperContext::encoder_window_seconds() const {
  return audio_processor_->config().encoder_window_seconds;
}

// ============================================================================
// Public API — thin wrappers delegating to template methods
// ============================================================================

std::vector<tcim::Tensor> WhisperContext::RunEncoder(
    const std::vector<float16>& mel_features, int n_mels, int n_frames) {
  (void)n_mels;
  (void)n_frames;
  std::vector<float> mel_f(mel_features.size());
  for (size_t i = 0; i < mel_features.size(); ++i)
    mel_f[i] = static_cast<float>(mel_features[i]);
  do_encode(mel_f, n_mels, n_frames);
  return encoder_outputs_;
}

std::vector<float16> WhisperContext::Encode(
    const std::vector<float>& mel_features, int n_mels, int n_frames) {
  (void)mel_features;
  (void)n_mels;
  (void)n_frames;
  return {};
}

Token WhisperContext::DetectLanguage() {
  if (detected_lang_id_ != 0) return detected_lang_id_;
  return do_detect_language();
}

std::vector<Token> WhisperContext::BuildPrompt(Token language_token) {
  return {asr_model()->sot_token_id(), language_token,
          asr_model()->transcribe_token_id(),
          asr_model()->notimestamps_token_id()};
}

Token WhisperContext::prefill(const std::vector<Token>& tokens) {
  return do_prefill(tokens);
}

Token WhisperContext::decode(Token prev_token) { return do_decode(prev_token); }

// ============================================================================
// Profiling hook implementations
// ============================================================================

void WhisperContext::encode_preprocess_impl(const std::vector<float>& mel,
                                            int n_mels, int n_frames) {
  encode_n_mels_ = n_mels;
  encode_n_frames_ = n_frames;
  auto* model = static_cast<WhisperModel*>(asr_model());

  std::vector<float16> mel_f16(mel.size());
  for (size_t i = 0; i < mel.size(); ++i)
    mel_f16[i] = static_cast<float16>(mel[i]);

  auto& encoder_input_map = model->encoder_input_map();
  std::string input_name = model->encoder_module()->GetInputName(0);
  auto& tensor = encoder_input_map[input_name];
  CHECK_TCIM_RET_STATUS(
      tensor.Buffer().CopyFromHost(mel_f16.data(), tensor.MemSize()));
  model->encoder_module()->SetInput(input_name, tensor);
}

void WhisperContext::encode_inference_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  model->encoder_module()->Run();
  model->encoder_module()->Sync();
}

void WhisperContext::encode_postprocess_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  encoder_outputs_.clear();
  int output_num = model->encoder_module()->GetOutputNum();
  auto& decode_input_map = model->decode_input_map();
  int num_decode_layers = model->num_decode_layers();
  int base_idx = model->base_idx();

  for (int idx = 0; idx < output_num; idx++) {
    std::string output_name = model->encoder_module()->GetOutputName(idx);
    int decoder_input_idx;
    if (idx < num_decode_layers) {
      decoder_input_idx = base_idx + 2 * num_decode_layers + idx;
    } else {
      decoder_input_idx =
          base_idx + 3 * num_decode_layers + (idx - num_decode_layers);
    }
    std::string decoder_name =
        model->decode_module()->GetInputName(decoder_input_idx);
    auto& output_tensor = decode_input_map[decoder_name];
    model->encoder_module()->GetOutput(output_name).CastTo(output_tensor);
    encoder_outputs_.push_back(output_tensor);
  }
}

void WhisperContext::detect_lang_preprocess_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  auto* decode_module = model->decode_module().get();
  auto& decode_input_map = model->decode_input_map();
  int num_heads = model->num_heads();
  int cache_max_len = model->cache_max_len();
  int num_decode_layers = model->num_decode_layers();
  int base_idx = model->base_idx();
  int sot_id = static_cast<int>(model->sot_token_id());

  for (int i = 0; i < base_idx; ++i) {
    std::string name = decode_module->GetInputName(i);
    auto& tensor = decode_input_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_ids") != std::string::npos) {
      std::vector<int> data = {sot_id};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("cache_position") != std::string::npos) {
      std::vector<int> data = {0};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("past_len") != std::string::npos) {
      std::vector<int> data = {0};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("current_len") != std::string::npos) {
      std::vector<int> data = {1};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("mask_attn") != std::string::npos) {
      std::vector<float16> mask(sz / sizeof(float16),
                                static_cast<float16>(-65504.0f));
      for (int j = 0; j < num_heads; ++j) {
        mask[j * cache_max_len] = static_cast<float16>(0.0f);
      }
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(mask.data(), sz));
    } else if (name.find("encoder_attention_mask") != std::string::npos) {
      std::vector<float16> enc_mask(sz / sizeof(float16),
                                    static_cast<float16>(0.0f));
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(enc_mask.data(), sz));
    }
    decode_module->SetInput(name, tensor);
  }

  for (int l = 0; l < num_decode_layers; ++l) {
    int key_idx = base_idx + 2 * num_decode_layers + l;
    int val_idx = base_idx + 3 * num_decode_layers + l;
    decode_module->SetInput(decode_module->GetInputName(key_idx),
                            encoder_outputs_[l]);
    decode_module->SetInput(decode_module->GetInputName(val_idx),
                            encoder_outputs_[num_decode_layers + l]);
  }
}

void WhisperContext::detect_lang_inference_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  model->decode_module()->Run();
  model->decode_module()->Sync();
}

Token WhisperContext::detect_lang_postprocess_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  auto* decode_module = model->decode_module().get();
  auto tokenizer = model->tokenizer();
  auto out_name = decode_module->GetOutputName(0);
  auto dev_output = decode_module->GetDevOutput(out_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());
  int vocab_size = decode_module->GetOutputInfo(out_name).Shape()[2];

  const auto& lang_ids = model->lang_to_id();

  std::vector<float16> masked_logits(logits, logits + vocab_size);
  for (int i = 0; i < vocab_size; ++i) {
    bool is_non_lang = true;
    for (Token id : lang_ids) {
      if (static_cast<int>(id) == i) {
        is_non_lang = false;
        break;
      }
    }
    if (is_non_lang) {
      masked_logits[i] = static_cast<float16>(-65504.0f);
    }
  }

  detected_lang_id_ = static_cast<Token>(
      houmo::eigen_argmax<float16>(masked_logits.data(), vocab_size));
  if (detected_lang_id_ == 0) {
    detected_lang_id_ = model->lang_token_id("zh");
  }
  return detected_lang_id_;
}

void WhisperContext::prefill_preprocess_impl(const std::vector<Token>& tokens) {
  auto* model = static_cast<WhisperModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto& prefill_input_map = model->prefill_input_map();
  int num_heads = model->num_heads();
  int cache_max_len = model->cache_max_len();
  int num_decode_layers = model->num_decode_layers();
  int base_idx = model->base_idx();
  int prompt_len = static_cast<int>(tokens.size());
  prefill_seq_len_ = prompt_len;
  prefill_tokens_ = tokens;

  for (int i = 0; i < base_idx; ++i) {
    std::string name = prefill_module->GetInputName(i);
    auto& tensor = prefill_input_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_ids") != std::string::npos) {
      std::vector<int> data;
      for (Token t : tokens) data.push_back(static_cast<int>(t));
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("cache_position") != std::string::npos) {
      std::vector<int> data;
      for (int p = 0; p < prompt_len; ++p) data.push_back(p);
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("past_len") != std::string::npos) {
      std::vector<int> data = {0};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("current_len") != std::string::npos) {
      std::vector<int> data = {prompt_len};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("mask_attn") != std::string::npos) {
      std::vector<float16> mask(sz / sizeof(float16),
                                static_cast<float16>(-65504.0f));
      for (int b = 0; b < num_heads; ++b) {
        for (int r = 0; r < prompt_len; ++r) {
          for (int c = 0; c <= r && c < cache_max_len; ++c) {
            mask[b * prompt_len * cache_max_len + r * cache_max_len + c] =
                static_cast<float16>(0.0f);
          }
        }
      }
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(mask.data(), sz));
    } else if (name.find("encoder_attention_mask") != std::string::npos) {
      std::vector<float16> enc_mask(sz / sizeof(float16),
                                    static_cast<float16>(0.0f));
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(enc_mask.data(), sz));
    }
    prefill_module->SetInput(name, tensor);
  }

  for (int l = 0; l < num_decode_layers; ++l) {
    int key_idx = base_idx + 2 * num_decode_layers + l;
    int val_idx = base_idx + 3 * num_decode_layers + l;
    prefill_module->SetInput(prefill_module->GetInputName(key_idx),
                             encoder_outputs_[l]);
    prefill_module->SetInput(prefill_module->GetInputName(val_idx),
                             encoder_outputs_[num_decode_layers + l]);
  }
}

void WhisperContext::prefill_inference_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  model->prefill_module()->Run();
  model->prefill_module()->Sync();
}

Token WhisperContext::prefill_postprocess_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto* decode_module = model->decode_module().get();
  auto tokenizer = model->tokenizer();
  int num_decode_layers = model->num_decode_layers();
  int base_idx = model->base_idx();
  int prompt_len = prefill_seq_len_;

  for (int i = 0; i < 2 * num_decode_layers; ++i) {
    auto out_name = prefill_module->GetOutputName(i + 1);
    auto cache = prefill_module->GetDevOutput(out_name);
    decode_module->SetDevInput(decode_module->GetInputName(base_idx + i),
                               cache);
    decode_module->SetDevOutput(decode_module->GetOutputName(i + 1), cache);
  }
  for (int i = 0; i < 2 * num_decode_layers; ++i) {
    auto enc_kv = prefill_module->GetDevInput(
        prefill_module->GetInputName(base_idx + 2 * num_decode_layers + i));
    decode_module->SetDevInput(
        decode_module->GetInputName(base_idx + 2 * num_decode_layers + i),
        enc_kv);
  }

  auto out_name = prefill_module->GetOutputName(0);
  auto dev_output = prefill_module->GetDevOutput(out_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());

  int vocab_size = prefill_module->GetOutputInfo(out_name).Shape()[2];

  if (!sampler()) set_sampler(SamplingParams{});
  Token first_token =
      sampler()->sample(logits + (prompt_len - 1) * vocab_size, vocab_size);
  generated_ids_.push_back(first_token);
  decode_position_ = prompt_len;
  context_length_ = prompt_len;

  return first_token;
}

void WhisperContext::decode_preprocess_impl(Token prev_token) {
  auto* model = static_cast<WhisperModel*>(asr_model());
  auto* decode_module = model->decode_module().get();
  auto& decode_input_map = model->decode_input_map();
  int num_heads = model->num_heads();
  int cache_max_len = model->cache_max_len();
  int base_idx = model->base_idx();
  int pos = decode_position_;

  for (int i = 0; i < base_idx; ++i) {
    std::string name = decode_module->GetInputName(i);
    auto& tensor = decode_input_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_ids") != std::string::npos) {
      std::vector<int> data = {static_cast<int>(prev_token)};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("cache_position") != std::string::npos) {
      std::vector<int> data = {pos};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("past_len") != std::string::npos) {
      std::vector<int> data = {pos};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("current_len") != std::string::npos) {
      std::vector<int> data = {1};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("mask_attn") != std::string::npos) {
      std::vector<float16> mask(sz / sizeof(float16),
                                static_cast<float16>(0.0f));
      for (int b = 0; b < num_heads; ++b) {
        for (int c = pos + 1; c < cache_max_len; ++c) {
          mask[b * cache_max_len + c] = static_cast<float16>(-65504.0f);
        }
      }
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(mask.data(), sz));
    } else if (name.find("encoder_attention_mask") != std::string::npos) {
      std::vector<float16> enc_mask(sz / sizeof(float16),
                                    static_cast<float16>(0.0f));
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(enc_mask.data(), sz));
    }
    decode_module->SetInput(name, tensor);
  }
}

void WhisperContext::decode_inference_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  model->decode_module()->Run();
  model->decode_module()->Sync();
}

Token WhisperContext::decode_postprocess_impl() {
  auto* model = static_cast<WhisperModel*>(asr_model());
  auto* decode_module = model->decode_module().get();
  auto tokenizer = model->tokenizer();

  auto out_name = decode_module->GetOutputName(0);
  auto dev_output = decode_module->GetDevOutput(out_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());

  int vocab_size = decode_module->GetOutputInfo(out_name).Shape()[2];

  Token next_token = sampler()->sample(logits, vocab_size, generated_ids_);
  generated_ids_.push_back(next_token);
  decode_position_++;
  context_length_++;

  return next_token;
}

// ============================================================================
// Transcribe — with top-level profiling
// ============================================================================

void WhisperContext::Transcribe(const std::string& audio_path,
                                const SamplingParams& params,
                                ASRTokenCallback callback) {
  auto* model = static_cast<WhisperModel*>(asr_model());
  profiler_.reset();
  profiler_.set_root_stage("transcribe");
  auto& p = profiler_;

  set_sampler(params);

  if (!audio_processor_) {
    std::cerr << "AudioProcessor not initialized" << std::endl;
    return;
  }

  p.start("transcribe");

  std::vector<AudioData> chunks;
  float total_duration = 0.0f;
  {
    auto t = p.scope("transcribe.audio_load");
    auto audio = audio_processor_->LoadAudio(audio_path);
    if (audio.pcm.empty()) {
      std::cerr << "No audio loaded" << std::endl;
      p.stop("transcribe");
      return;
    }
    total_duration = static_cast<float>(audio.duration);
    chunks = audio_processor_->ChunkPCM(audio);
  }
  if (chunks.empty()) {
    std::cerr << "No audio chunks extracted" << std::endl;
    p.stop("transcribe");
    return;
  }

  bool first_chunk = true;
  int max_tokens = params.max_tokens > 0 ? params.max_tokens : 448;
  Token eos_id = model->eos_token_ids()[0];

  for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    MelFeatures features;
    {
      auto t = p.scope("transcribe.feature_extract");
      features = audio_processor_->ExtractFeatures(chunks[chunk_idx]);
    }
    if (features.data.empty()) break;

    std::vector<float> mel_f(features.data.size());
    for (size_t i = 0; i < features.data.size(); ++i)
      mel_f[i] = static_cast<float>(features.data[i]);
    do_encode(mel_f, features.feature_dim, features.num_frames);

    Token lang_id;
    if (first_chunk) {
      if (language_ == "auto") {
        lang_id = do_detect_language();
      } else {
        lang_id = model->lang_token_id(language_);
      }
      first_chunk = false;
    } else {
      lang_id = detected_lang_id_;
    }

    auto prompt = BuildPrompt(lang_id);
    reset();

    Token first_token = do_prefill(prompt);
    // Bugfix: count prefill input tokens cumulatively across chunks
    p.set_input_tokens(p.input_tokens() + prefill_seq_len_);
    // Bugfix: only record TTFT on first chunk
    if (chunk_idx == 0) {
      p.record_ttft();
    }

    if (callback) callback(first_token);

    int step = 0;
    Token prev_token = first_token;

    while (step < max_tokens && prev_token != eos_id) {
      Token next = do_decode(prev_token);
      p.add_output_token();
      step++;

      if (next == eos_id) break;

      if (callback) callback(next);
      prev_token = next;
    }
  }

  p.stop("transcribe");
  fill_perf_info(total_duration);
}

void WhisperContext::set_language(const std::string& language) {
  language_ = language;
}

// Static registration for Whisper ASR model
REGISTER_MODEL(
    ASRModel, whisper_asr, ModelSeries::kWhisperASR,
    [](const ModelConfig& c) { return std::make_unique<WhisperModel>(c); },
    "Whisper ASR model");

}  // namespace houmo
