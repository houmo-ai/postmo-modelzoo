/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: glm_asr_model.cc
 * Description:
 *   GLM-ASR model implementation
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

#include "glm_asr_model.h"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <limits>

#include "base/tcim_utils.h"
#include "core/model_factory.h"

namespace fs = std::filesystem;

namespace houmo {

namespace {

static int compute_audio_output_length(int valid_frames) {
  int L = valid_frames;
  L = (L + 2 * 1 - (3 - 1) - 1) / 1 + 1;
  L = (L + 2 * 1 - (3 - 1) - 1) / 2 + 1;
  int merge_factor = 4;
  return (L - merge_factor) / merge_factor + 1;
}

}  // namespace

// ============================================================================
// GlmAsrModel Implementation
// ============================================================================

GlmAsrModel::GlmAsrModel(const ModelConfig& config) : ASRModel(config) {
  load();
}

GlmAsrModel::~GlmAsrModel() {
  encoder_module_.reset();
  prefill_module_.reset();
  decode_module_.reset();
}

void GlmAsrModel::load() {
  dev_manager_ = std::make_unique<tcim::DevManager>(
      tcim::DevManager::Create(config_.devices));
  weight_manager_ = std::make_unique<tcim::Module::WeightManager>(
      tcim::Module::WeightManager::CreateWeightManager(*dev_manager_));

  // Load encode
  {
    const std::string& encoder_path = config_.extra_params.at("encoder_path");
    auto encoder_option = tcim::Module::Option(*weight_manager_);
    encoder_option.EnableIOLazyMode(true);
    encoder_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        encoder_module_->LoadModel(encoder_path, encoder_option));

    auto input0_shape =
        encoder_module_->GetInputInfo(encoder_module_->GetInputName(0)).Shape();
    if (input0_shape.size() >= 2) {
      n_mels_ = input0_shape[1];
      n_frames_ = input0_shape[2];
    }
    std::cout << "Detected n_mels=" << n_mels_ << " n_frames=" << n_frames_
              << std::endl;
  }

  // Load prefill
  {
    auto prefill_option = tcim::Module::Option(*weight_manager_);
    prefill_option.EnableIOLazyMode(true);
    prefill_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        prefill_module_->LoadModel(config_.prefill_path, prefill_option));

    auto emb_shape =
        prefill_module_->GetInputInfo(prefill_module_->GetInputName(0)).Shape();
    if (emb_shape.size() >= 2) {
      max_prefill_ = emb_shape[1];
      hidden_size_ = emb_shape[2];
    }
    num_decode_layers_ = 0;
    for (int i = 0; i < prefill_module_->GetInputNum(); ++i) {
      auto name = prefill_module_->GetInputName(i);
      if (name.find("kcache") != std::string::npos) ++num_decode_layers_;
    }
    std::cout << "Prefill: max_prefill=" << max_prefill_
              << " hidden_size=" << hidden_size_
              << " layers=" << num_decode_layers_
              << " max_new_tokens=" << max_new_tokens_ << std::endl;
  }

  // Load decode
  {
    auto decode_option = tcim::Module::Option(*weight_manager_);
    decode_option.EnableIOLazyMode(true);
    decode_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        decode_module_->LoadModel(config_.decode_path, decode_option));
    std::cout << "Decode model loaded" << std::endl;
  }

  // Share KV cache between prefill and decode
  for (int i = 3; i < prefill_module_->GetInputNum(); ++i) {
    auto name = prefill_module_->GetInputName(i);
    auto dev_cache = prefill_module_->GetDevInput(name);
    decode_module_->SetDevInput(name, dev_cache);
    if (i - 2 < decode_module_->GetOutputNum()) {
      decode_module_->SetDevOutput(decode_module_->GetOutputName(i - 2),
                                   dev_cache);
    }
  }

  // Init input maps
  {
    for (int i = 0; i < encoder_module_->GetInputNum(); ++i) {
      auto name = encoder_module_->GetInputName(i);
      auto info = encoder_module_->GetInputInfo(name).AsContiguous();
      encoder_input_map_[name] = tcim::Tensor::CreateHostTensor(info);
    }
    for (int i = 0; i < prefill_module_->GetInputNum(); ++i) {
      auto name = prefill_module_->GetInputName(i);
      auto info = prefill_module_->GetInputInfo(name).AsContiguous();
      prefill_input_map_[name] = tcim::Tensor::CreateHostTensor(info);
    }
    for (int i = 0; i < decode_module_->GetInputNum(); ++i) {
      auto name = decode_module_->GetInputName(i);
      auto info = decode_module_->GetInputInfo(name).AsContiguous();
      decode_input_map_[name] = tcim::Tensor::CreateHostTensor(info);
    }
    std::cout << "Input tensors initialized" << std::endl;
  }

  // Load embedding
  {
    std::string embed_path = config_.embedding_path;
    embedding_ =
        std::make_unique<Embedding>(embed_path, hidden_size_, max_prefill_);
    std::cout << "Embedding loaded: " << embed_path << std::endl;
  }

  // Load tokenizer
  {
    if (fs::exists(config_.tokenizer_path)) {
      tokenizer_ = std::make_shared<HfTokenizer>(config_.tokenizer_path);
    }
  }

  // Init token IDs
  {
    audio_token_id_ = 59260;
    eos_token_ids_ = {59246, 59253, 59255};
    std::cout << "Token IDs: audio=" << audio_token_id_ << " eos=["
              << eos_token_ids_[0] << "," << eos_token_ids_[1] << ","
              << eos_token_ids_[2] << "]" << std::endl;
  }
}

std::unique_ptr<Context> GlmAsrModel::create_context(int n_ctx) {
  if (n_ctx <= 0) n_ctx = max_prefill_;
  return std::make_unique<GlmAsrContext>(this, n_ctx);
}

const float16* GlmAsrModel::get_embedding(Token token) const {
  return embedding_->token_embedding(token);
}

const float16* GlmAsrModel::get_embedding(
    const std::vector<Token>& tokens) const {
  return embedding_->token_embedding(tokens);
}

// ============================================================================
// GlmAsrContext Implementation
// ============================================================================

GlmAsrContext::GlmAsrContext(ASRModel* model, int n_ctx)
    : ASRContext(model, n_ctx),
      audio_processor_(std::make_shared<AudioProcessor>(
          AudioProcessorConfig{.sample_rate = 16000,
                               .n_mels = model->n_mels(),
                               .chunk_seconds = 30,
                               .encoder_window_seconds = 30})) {
  std::cout << "GlmAsrContext created, n_mels=" << model->n_mels() << std::endl;
}

void GlmAsrContext::set_audio_processor(int sample_rate, int chunk_seconds,
                                        int encoder_window_seconds) {
  auto cfg = audio_processor_->config();
  cfg.sample_rate = sample_rate;
  cfg.chunk_seconds = chunk_seconds;
  cfg.encoder_window_seconds = encoder_window_seconds;
  audio_processor_ = std::make_shared<AudioProcessor>(cfg);
}

int GlmAsrContext::sample_rate() const {
  return audio_processor_->sample_rate();
}

int GlmAsrContext::chunk_seconds() const {
  return audio_processor_->config().chunk_seconds;
}

int GlmAsrContext::encoder_window_seconds() const {
  return audio_processor_->config().encoder_window_seconds;
}

// ============================================================================
// Public API — thin wrappers
// ============================================================================

std::vector<float16> GlmAsrContext::Encode(
    const std::vector<float>& mel_features, int n_mels, int n_frames) {
  do_encode(mel_features, n_mels, n_frames);
  return audio_embeds_;
}

Token GlmAsrContext::DetectLanguage() { return 0; }

std::vector<Token> GlmAsrContext::BuildPrompt(Token language_token) {
  (void)language_token;
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  auto tokenizer = model->tokenizer();

  Token user_id = tokenizer->token_to_id("<|user|>");
  Token begin_audio_id = tokenizer->token_to_id("<|begin_of_audio|>");
  Token end_audio_id = tokenizer->token_to_id("<|end_of_audio|>");
  Token assistant_id = tokenizer->token_to_id("<|assistant|>");
  Token audio_token = model->audio_token_id();
  int T_out = static_cast<int>(audio_embeds_.size()) / model->hidden_size();

  std::vector<Token> tokens;
  tokens.push_back(user_id);
  tokens.push_back(10);
  tokens.push_back(begin_audio_id);
  for (int i = 0; i < T_out; ++i) tokens.push_back(audio_token);
  tokens.push_back(end_audio_id);
  tokens.push_back(user_id);
  tokens.push_back(10);
  auto text =
      tokenizer->encode("Please transcribe this audio into text", false, false);
  tokens.insert(tokens.end(), text.begin(), text.end());
  tokens.push_back(assistant_id);
  tokens.push_back(10);

  return tokens;
}

Token GlmAsrContext::prefill(const std::vector<Token>& tokens) {
  return do_prefill(tokens);
}

Token GlmAsrContext::decode(Token prev_token) { return do_decode(prev_token); }

// ============================================================================
// Profiling hook implementations
// ============================================================================

void GlmAsrContext::encode_preprocess_impl(const std::vector<float>& mel,
                                           int n_mels, int n_frames) {
  encode_n_frames_ = n_frames;
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  int max_frames = model->n_frames();
  int data_frames = static_cast<int>(mel.size()) / n_mels;

  std::vector<float> padded(max_frames * n_mels, 0.0f);
  int copy_frames = std::min(n_frames, max_frames);
  for (int m = 0; m < n_mels; ++m) {
    for (int f = 0; f < copy_frames; ++f) {
      padded[m * max_frames + f] = mel[m * data_frames + f];
    }
  }

  auto& encoder_input_map = model->encoder_input_map();
  std::string input_name = model->encoder_module()->GetInputName(0);
  auto& tensor = encoder_input_map[input_name];
  CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(
      padded.data(), padded.size() * sizeof(float)));
  model->encoder_module()->SetInput(input_name, tensor);
}

void GlmAsrContext::encode_inference_impl() {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  model->encoder_module()->Run();
  model->encoder_module()->Sync();
}

void GlmAsrContext::encode_postprocess_impl() {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  int hidden_size = model->hidden_size();

  std::string out_name = model->encoder_module()->GetOutputName(0);
  auto out_info =
      model->encoder_module()->GetOutputInfo(out_name).AsContiguous();
  tcim::Tensor out_tensor = tcim::Tensor::CreateHostTensor(out_info);
  model->encoder_module()->GetOutput(out_name).CastTo(out_tensor);

  size_t out_byte_size = out_tensor.MemSize();
  size_t elem_size = sizeof(float);
  if (out_byte_size % sizeof(float) == 0 &&
      out_byte_size / sizeof(float) % hidden_size == 0) {
    elem_size = sizeof(float);
  } else {
    elem_size = sizeof(float16);
  }
  int encoder_frames =
      static_cast<int>(out_byte_size / elem_size / hidden_size);
  int T_out = compute_audio_output_length(encode_n_frames_);
  int total_elems = std::min(T_out, encoder_frames) * hidden_size;

  audio_embeds_.resize(total_elems);
  if (elem_size == sizeof(float)) {
    float* raw = static_cast<float*>(out_tensor.Buffer().Data());
    for (size_t i = 0; i < audio_embeds_.size(); ++i) {
      audio_embeds_[i] = static_cast<float16>(raw[i]);
    }
  } else {
    float16* raw = static_cast<float16*>(out_tensor.Buffer().Data());
    std::copy(raw, raw + total_elems, audio_embeds_.begin());
  }
}

void GlmAsrContext::prefill_preprocess_impl(const std::vector<Token>& tokens) {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto& prefill_input_map = model->prefill_input_map();
  int max_prefill = model->max_prefill();
  int hidden_size = model->hidden_size();

  const float16* text_embeds = model->get_embedding(tokens);
  int audio_token = static_cast<int>(model->audio_token_id());
  int audio_embed_len = static_cast<int>(audio_embeds_.size()) / hidden_size;

  std::vector<float16> fused(max_prefill * hidden_size,
                             static_cast<float16>(0.0f));
  int cursor = 0;
  int audio_idx = 0;

  for (size_t i = 0; i < tokens.size() && cursor < max_prefill; ++i) {
    if (static_cast<int>(tokens[i]) == audio_token &&
        audio_idx < audio_embed_len) {
      std::copy(audio_embeds_.begin() + audio_idx * hidden_size,
                audio_embeds_.begin() + (audio_idx + 1) * hidden_size,
                fused.begin() + cursor * hidden_size);
      audio_idx++;
      cursor++;
    } else {
      std::copy(text_embeds + i * hidden_size,
                text_embeds + (i + 1) * hidden_size,
                fused.begin() + cursor * hidden_size);
      cursor++;
    }
  }
  prefill_seq_len_ = cursor;

  for (int i = 0; i < prefill_module->GetInputNum(); ++i) {
    std::string name = prefill_module->GetInputName(i);
    auto& tensor = prefill_input_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_embeds") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(
          fused.data(), fused.size() * sizeof(float16)));
    } else if (name.find("valid_length") != std::string::npos) {
      std::vector<int> data = {0};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("current_length") != std::string::npos) {
      std::vector<int> data = {prefill_seq_len_};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else {
      continue;
    }
    prefill_module->SetInput(name, tensor);
  }
}

void GlmAsrContext::prefill_inference_impl() {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  model->prefill_module()->Run();
  model->prefill_module()->Sync();
}

Token GlmAsrContext::prefill_postprocess_impl() {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto* decode_module = model->decode_module().get();
  auto tokenizer = model->tokenizer();

  for (int i = 3; i < prefill_module->GetInputNum(); ++i) {
    auto name = prefill_module->GetInputName(i);
    auto dev_cache = prefill_module->GetDevInput(name);
    decode_module->SetDevInput(name, dev_cache);
  }

  auto out_name = prefill_module->GetOutputName(0);
  auto dev_output = prefill_module->GetDevOutput(out_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());
  int vocab_size = prefill_module->GetOutputInfo(out_name).Shape()[1];
  Token first_token =
      static_cast<Token>(houmo::eigen_argmax<float16>(logits, vocab_size));

  generated_ids_.push_back(first_token);
  decode_position_ = prefill_seq_len_;
  context_length_ = prefill_seq_len_;
  return first_token;
}

void GlmAsrContext::decode_preprocess_impl(Token prev_token) {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  auto* decode_module = model->decode_module().get();
  auto& decode_input_map = model->decode_input_map();

  const float16* emb = model->get_embedding(prev_token);
  int hidden_size = model->hidden_size();
  int pos = decode_position_;

  for (int i = 0; i < decode_module->GetInputNum(); ++i) {
    std::string name = decode_module->GetInputName(i);
    auto& tensor = decode_input_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_embeds") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(emb, hidden_size * sizeof(float16)));
    } else if (name.find("valid_length") != std::string::npos) {
      std::vector<int> data = {pos};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else if (name.find("current_length") != std::string::npos) {
      std::vector<int> data = {1};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data.data(), sz));
    } else {
      continue;
    }
    decode_module->SetInput(name, tensor);
  }
}

void GlmAsrContext::decode_inference_impl() {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  model->decode_module()->Run();
  model->decode_module()->Sync();
}

Token GlmAsrContext::decode_postprocess_impl() {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  auto* decode_module = model->decode_module().get();
  auto tokenizer = model->tokenizer();

  auto out_name = decode_module->GetOutputName(0);
  auto dev_output = decode_module->GetDevOutput(out_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());
  int vocab_size = decode_module->GetOutputInfo(out_name).Shape()[1];
  Token next_token =
      static_cast<Token>(houmo::eigen_argmax<float16>(logits, vocab_size));

  generated_ids_.push_back(next_token);
  decode_position_++;
  context_length_++;
  return next_token;
}

// ============================================================================
// Transcribe — with top-level profiling
// ============================================================================

void GlmAsrContext::Transcribe(const std::string& audio_path,
                               const SamplingParams& params,
                               ASRTokenCallback callback) {
  auto* model = static_cast<GlmAsrModel*>(asr_model());
  profiler_.reset();
  profiler_.set_root_stage("transcribe");
  auto& p = profiler_;

  p.start("transcribe");

  AudioData audio;
  std::vector<AudioData> chunks;
  float total_duration = 0.0f;
  {
    auto t = p.scope("transcribe.audio_load");
    audio = audio_processor_->LoadAudio(audio_path);
    chunks = audio_processor_->ChunkPCM(audio);
    total_duration = static_cast<float>(audio.duration);
    if (chunks.empty()) {
      p.stop("transcribe");
      return;
    }
  }

  const auto& eos_ids = model->eos_token_ids();

  for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
    const auto& chunk = chunks[chunk_idx];
    auto features = audio_processor_->ExtractFeatures(chunk);

    int actual_samples = static_cast<int>(chunk.pcm.size());
    int actual_frames = actual_samples / 160;
    if (actual_frames < 1) actual_frames = 1;

    do_encode(std::vector<float>(features.data.begin(), features.data.end()),
              features.feature_dim, actual_frames);

    auto prompt = BuildPrompt(0);
    reset();

    Token first_token = do_prefill(prompt);
    p.set_input_tokens(p.input_tokens() + static_cast<int>(prompt.size()));
    if (chunk_idx == 0) {
      p.record_ttft();
    }

    if (callback) callback(first_token);

    int max_tokens =
        params.max_tokens > 0 ? params.max_tokens : model->max_new_tokens();
    Token prev_token = first_token;
    int step = 0;

    while (step < max_tokens && prev_token != TokenNull) {
      Token next = do_decode(prev_token);
      p.add_output_token();
      step++;
      if (std::find(eos_ids.begin(), eos_ids.end(), next) != eos_ids.end()) {
        break;
      }
      if (callback) callback(next);
      prev_token = next;
    }
  }

  p.stop("transcribe");
  fill_perf_info(total_duration);
}

void GlmAsrContext::set_language(const std::string& language) {
  language_ = language;
}

REGISTER_MODEL(
    ASRModel, glm_asr, ModelSeries::kGlmAsr,
    [](const ModelConfig& c) { return std::make_unique<GlmAsrModel>(c); },
    "GLM-ASR model");

}  // namespace houmo
