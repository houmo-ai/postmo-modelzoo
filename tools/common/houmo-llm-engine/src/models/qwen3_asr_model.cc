/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_asr_model.cc
 * Description:
 *   Qwen3-ASR model implementation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SPDX-License-Identifier: Apache-2.0
 */

#include "models/qwen3_asr_model.h"

#include <filesystem>
#include <iostream>
#include <limits>

#include "base/tcim_utils.h"
#include "core/model_factory.h"

namespace fs = std::filesystem;

namespace houmo {

namespace {

static int compute_feat_extract_output_lengths(int input_lengths) {
  int feat_len = ((input_lengths % 100) - 1) / 2 + 1;
  int out = ((feat_len - 1) / 2 + 1 - 1) / 2 + 1 + (input_lengths / 100) * 13;
  return out;
}

}  // namespace

// ============================================================================
// Qwen3AsrModel
// ============================================================================

Qwen3AsrModel::Qwen3AsrModel(const ModelConfig& config) : ASRModel(config) {
  load();
}

Qwen3AsrModel::~Qwen3AsrModel() {
  encoder_module_.reset();
  prefill_module_.reset();
  decode_module_.reset();
}

void Qwen3AsrModel::load() {
  dev_manager_ = std::make_unique<tcim::DevManager>(
      tcim::DevManager::Create(config_.devices));
  weight_manager_ = std::make_unique<tcim::Module::WeightManager>(
      tcim::Module::WeightManager::CreateWeightManager(*dev_manager_));

  // Load encode
  {
    const std::string& encoder_path = config_.extra_params.at("encoder_path");
    auto opt = tcim::Module::Option(*weight_manager_);
    opt.EnableIOLazyMode(true);
    encoder_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(encoder_module_->LoadModel(encoder_path, opt));
    std::cout << "Encoder loaded: " << encoder_path << std::endl;

    auto s0 =
        encoder_module_->GetInputInfo(encoder_module_->GetInputName(0)).Shape();
    if (s0.size() >= 2) {
      n_mels_ = s0[1];
      max_feature_one_loop_ = s0[2];
    }
    std::cout << "Detected n_mels=" << n_mels_
              << " max_feature_one_loop=" << max_feature_one_loop_ << std::endl;
  }

  // Load prefill
  {
    auto opt = tcim::Module::Option(*weight_manager_);
    opt.EnableIOLazyMode(true);
    prefill_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        prefill_module_->LoadModel(config_.prefill_path, opt));

    auto s0 =
        prefill_module_->GetInputInfo(prefill_module_->GetInputName(0)).Shape();
    if (s0.size() >= 2) {
      max_prefill_ = s0[1];
      hidden_size_ = s0[2];
    }
    max_new_tokens_ = 2048;
    num_decode_layers_ = 0;
    for (int i = 0; i < prefill_module_->GetInputNum(); ++i) {
      auto name = prefill_module_->GetInputName(i);
      if (name.find("kcache") != std::string::npos) ++num_decode_layers_;
    }
    std::cout << "Prefill: max_prefill=" << max_prefill_
              << " hidden=" << hidden_size_ << " layers=" << num_decode_layers_
              << " max_new_tokens=" << max_new_tokens_ << std::endl;
  }

  // Load decode
  {
    auto opt = tcim::Module::Option(*weight_manager_);
    opt.EnableIOLazyMode(true);
    decode_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(decode_module_->LoadModel(config_.decode_path, opt));
    std::cout << "Decode loaded" << std::endl;
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
    audio_pad_id_ = 151676;
    eos_token_id_ = tokenizer_->eos_token_id();
    std::cout << "Token IDs: audio_pad=" << audio_pad_id_
              << " eos=" << eos_token_id_ << std::endl;
  }
}

std::unique_ptr<Context> Qwen3AsrModel::create_context(int n_ctx) {
  if (n_ctx <= 0) n_ctx = max_prefill_;
  return std::make_unique<Qwen3AsrContext>(this, n_ctx);
}

const float16* Qwen3AsrModel::get_embedding(Token token) const {
  return embedding_->token_embedding(token);
}

const float16* Qwen3AsrModel::get_embedding(
    const std::vector<Token>& tokens) const {
  return embedding_->token_embedding(tokens);
}

// ============================================================================
// Qwen3AsrContext
// ============================================================================

Qwen3AsrContext::Qwen3AsrContext(ASRModel* model, int n_ctx)
    : ASRContext(model, n_ctx) {
  AudioProcessorConfig cfg;
  cfg.sample_rate = 16000;
  cfg.n_mels = model->n_mels();
  cfg.chunk_seconds = 30;
  cfg.encoder_window_seconds = 30;
  audio_processor_ = std::make_shared<AudioProcessor>(cfg);
  std::cout << "Qwen3AsrContext created n_mels=" << cfg.n_mels << std::endl;
}

void Qwen3AsrContext::set_audio_processor(int sample_rate, int chunk_seconds,
                                          int encoder_window_seconds) {
  auto cfg = audio_processor_->config();
  cfg.sample_rate = sample_rate;
  cfg.chunk_seconds = chunk_seconds;
  cfg.encoder_window_seconds = encoder_window_seconds;
  audio_processor_ = std::make_shared<AudioProcessor>(cfg);
}

// ============================================================================
// Public API — thin wrappers
// ============================================================================

std::vector<float16> Qwen3AsrContext::Encode(
    const std::vector<float>& mel_features, int n_mels, int n_frames) {
  do_encode(mel_features, n_mels, n_frames);
  return audio_embeds_;
}

Token Qwen3AsrContext::DetectLanguage() { return 0; }

std::vector<Token> Qwen3AsrContext::BuildPrompt(Token language_token) {
  (void)language_token;
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  auto tokenizer = model->tokenizer();
  std::vector<Token> tokens;

  auto sys = tokenizer->encode(
      "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", false,
      false);
  tokens.insert(tokens.end(), sys.begin(), sys.end());

  auto user_start = tokenizer->encode("<|im_start|>user\n", false, false);
  tokens.insert(tokens.end(), user_start.begin(), user_start.end());

  tokens.push_back(static_cast<Token>(151669));

  tokens.push_back(model->audio_pad_id());

  tokens.push_back(static_cast<Token>(151670));

  auto user_end = tokenizer->encode("<|im_end|>\n", false, false);
  tokens.insert(tokens.end(), user_end.begin(), user_end.end());

  auto asst = tokenizer->encode("<|im_start|>assistant\n", false, false);
  tokens.insert(tokens.end(), asst.begin(), asst.end());

  return tokens;
}

Token Qwen3AsrContext::prefill(const std::vector<Token>& tokens) {
  return do_prefill(tokens);
}

Token Qwen3AsrContext::decode(Token prev_token) {
  return do_decode(prev_token);
}

// ============================================================================
// Profiling hook implementations
// ============================================================================

void Qwen3AsrContext::encode_preprocess_impl(const std::vector<float>& mel,
                                             int n_mels, int n_frames) {
  encode_n_mels_ = n_mels;
  encode_n_frames_ = n_frames;
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  int max_loop = model->max_feature_per_loop();

  std::vector<float> padded(max_loop * n_mels, 0.0f);
  int data_frames = static_cast<int>(mel.size()) / n_mels;
  int copy_frames = std::min(n_frames, max_loop);
  for (int m = 0; m < n_mels; ++m) {
    for (int f = 0; f < copy_frames; ++f) {
      padded[m * max_loop + f] = mel[m * data_frames + f];
    }
  }

  auto& encoder_map = model->encoder_input_map();
  auto* enc = model->encoder_module().get();

  std::vector<float16> padded_f16(padded.size());
  for (size_t i = 0; i < padded.size(); ++i)
    padded_f16[i] = static_cast<float16>(padded[i]);
  std::string feat_name = enc->GetInputName(0);
  auto& feat_tensor = encoder_map[feat_name];
  CHECK_TCIM_RET_STATUS(feat_tensor.Buffer().CopyFromHost(
      padded_f16.data(), padded_f16.size() * sizeof(float16)));
  enc->SetInput(feat_name, feat_tensor);

  if (enc->GetInputNum() > 1) {
    std::string len_name = enc->GetInputName(1);
    auto& len_tensor = encoder_map[len_name];
    std::vector<int> len_data = {n_frames};
    CHECK_TCIM_RET_STATUS(len_tensor.Buffer().CopyFromHost(
        len_data.data(), len_tensor.MemSize()));
    enc->SetInput(len_name, len_tensor);
  }
}

void Qwen3AsrContext::encode_inference_impl() {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  model->encoder_module()->Run();
  model->encoder_module()->Sync();
}

void Qwen3AsrContext::encode_postprocess_impl() {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  auto* enc = model->encoder_module().get();

  std::string out_name = enc->GetOutputName(0);
  auto out_info = enc->GetOutputInfo(out_name).AsContiguous();
  tcim::Tensor out_tensor = tcim::Tensor::CreateHostTensor(out_info);
  enc->GetOutput(out_name).CastTo(out_tensor);

  int T_out = compute_feat_extract_output_lengths(encode_n_frames_);
  int hidden = model->hidden_size();
  int total = T_out * hidden;
  float16* raw = static_cast<float16*>(out_tensor.Buffer().Data());

  audio_embeds_.assign(raw, raw + total);
}

void Qwen3AsrContext::prefill_preprocess_impl(
    const std::vector<Token>& tokens) {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto& prefill_map = model->prefill_input_map();
  int max_prefill = model->max_prefill();
  int hidden = model->hidden_size();

  const float16* text_embeds = model->get_embedding(tokens);
  int audio_embed_len = static_cast<int>(audio_embeds_.size()) / hidden;
  int audio_token = static_cast<int>(model->audio_pad_id());

  std::vector<float16> fused(max_prefill * hidden, static_cast<float16>(0.0f));
  int cursor = 0;

  for (size_t i = 0; i < tokens.size() && cursor < max_prefill; ++i) {
    if (static_cast<int>(tokens[i]) == audio_token) {
      int copy_n = std::min(audio_embed_len, max_prefill - cursor) * hidden;
      std::copy(audio_embeds_.begin(), audio_embeds_.begin() + copy_n,
                fused.begin() + cursor * hidden);
      cursor += audio_embed_len;
    } else {
      std::copy(text_embeds + i * hidden, text_embeds + (i + 1) * hidden,
                fused.begin() + cursor * hidden);
      cursor++;
    }
  }
  prefill_seq_len_ = cursor;

  for (int i = 0; i < prefill_module->GetInputNum(); ++i) {
    std::string name = prefill_module->GetInputName(i);
    auto& tensor = prefill_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_embeds") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(fused.data(), sz));
    } else if (name.find("valid_length") != std::string::npos) {
      std::vector<int> v = {0};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(v.data(), sz));
    } else if (name.find("current_length") != std::string::npos) {
      std::vector<int> v = {prefill_seq_len_};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(v.data(), sz));
    } else {
      continue;
    }
    prefill_module->SetInput(name, tensor);
  }
}

void Qwen3AsrContext::prefill_inference_impl() {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  model->prefill_module()->Run();
  model->prefill_module()->Sync();
}

Token Qwen3AsrContext::prefill_postprocess_impl() {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto* dec = model->decode_module().get();
  auto tokenizer = model->tokenizer();

  for (int i = 3; i < prefill_module->GetInputNum(); ++i) {
    auto name = prefill_module->GetInputName(i);
    dec->SetDevInput(name, prefill_module->GetDevInput(name));
  }

  std::string out_name = prefill_module->GetOutputName(0);
  auto out_info = prefill_module->GetOutputInfo(out_name).AsContiguous();
  tcim::Tensor out_tensor = tcim::Tensor::CreateHostTensor(out_info);
  prefill_module->GetOutput(out_name).CastTo(out_tensor);

  size_t out_bytes = out_tensor.MemSize();
  Token first_token = 0;
  float max_val = -std::numeric_limits<float>::infinity();
  int vocab = tokenizer->vocab_size();

  if (out_bytes == static_cast<size_t>(vocab) * sizeof(float)) {
    int count = static_cast<int>(out_bytes / sizeof(float));
    float* logits32 = static_cast<float*>(out_tensor.Buffer().Data());
    for (int i = 0; i < count; ++i) {
      float val = logits32[i];
      if (val > max_val) {
        max_val = val;
        first_token = static_cast<Token>(i);
      }
    }
  } else {
    int count = static_cast<int>(out_bytes / sizeof(float16));
    float16* logits16 = static_cast<float16*>(out_tensor.Buffer().Data());
    for (int i = 0; i < count; ++i) {
      float val = static_cast<float>(logits16[i]);
      if (val > max_val) {
        max_val = val;
        first_token = static_cast<Token>(i);
      }
    }
  }

  generated_ids_.push_back(first_token);
  decode_position_ = prefill_seq_len_;
  context_length_ = prefill_seq_len_;
  return first_token;
}

void Qwen3AsrContext::decode_preprocess_impl(Token prev_token) {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  auto* dec = model->decode_module().get();
  auto& dec_map = model->decode_input_map();
  int hidden = model->hidden_size();
  int pos = decode_position_;
  const float16* emb = model->get_embedding(prev_token);

  for (int i = 0; i < dec->GetInputNum(); ++i) {
    std::string name = dec->GetInputName(i);
    auto& tensor = dec_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_embeds") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(emb, sz));
    } else if (name.find("valid_length") != std::string::npos) {
      std::vector<int> v = {pos};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(v.data(), sz));
    } else if (name.find("current_length") != std::string::npos) {
      std::vector<int> v = {1};
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(v.data(), sz));
    } else {
      continue;
    }
    dec->SetInput(name, tensor);
  }
}

void Qwen3AsrContext::decode_inference_impl() {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  model->decode_module()->Run();
  model->decode_module()->Sync();
}

Token Qwen3AsrContext::decode_postprocess_impl() {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  auto* dec = model->decode_module().get();
  auto tokenizer = model->tokenizer();

  std::string out_name = dec->GetOutputName(0);
  auto out_info = dec->GetOutputInfo(out_name).AsContiguous();
  tcim::Tensor out_tensor = tcim::Tensor::CreateHostTensor(out_info);
  dec->GetOutput(out_name).CastTo(out_tensor);

  size_t out_bytes = out_tensor.MemSize();
  Token next_token = 0;
  float max_val = -std::numeric_limits<float>::infinity();
  int vocab = tokenizer->vocab_size();

  if (out_bytes == static_cast<size_t>(vocab) * sizeof(float)) {
    int count = static_cast<int>(out_bytes / sizeof(float));
    float* logits32 = static_cast<float*>(out_tensor.Buffer().Data());
    for (int i = 0; i < count; ++i) {
      float val = logits32[i];
      if (val > max_val) {
        max_val = val;
        next_token = static_cast<Token>(i);
      }
    }
  } else {
    int count = static_cast<int>(out_bytes / sizeof(float16));
    float16* logits16 = static_cast<float16*>(out_tensor.Buffer().Data());
    for (int i = 0; i < count; ++i) {
      float val = static_cast<float>(logits16[i]);
      if (val > max_val) {
        max_val = val;
        next_token = static_cast<Token>(i);
      }
    }
  }

  generated_ids_.push_back(next_token);
  decode_position_++;
  context_length_++;
  return next_token;
}

// ============================================================================
// Transcribe — with top-level profiling
// ============================================================================

void Qwen3AsrContext::Transcribe(const std::string& audio_path,
                                 const SamplingParams& params,
                                 ASRTokenCallback callback) {
  auto* model = static_cast<Qwen3AsrModel*>(asr_model());
  auto tokenizer = model->tokenizer();
  profiler_.reset();
  profiler_.set_root_stage("transcribe");
  auto& p = profiler_;

  asr_start = tokenizer->token_to_id("<asr_text>");

  p.start("transcribe");

  AudioData audio;
  MelFeatures features;
  float total_duration = 0.0f;
  {
    auto t = p.scope("transcribe.audio_load");
    audio = audio_processor_->LoadAudio(audio_path);
    if (audio.pcm.empty()) {
      p.stop("transcribe");
      return;
    }
    total_duration = static_cast<float>(audio.duration);
    int full_duration = static_cast<int>(audio.duration) + 1;
    auto full_cfg = audio_processor_->config();
    full_cfg.encoder_window_seconds = full_duration;
    full_cfg.chunk_seconds = full_duration;
    AudioProcessor full_proc(full_cfg);
    features = full_proc.ExtractFeatures(audio);
    if (features.data.empty()) {
      p.stop("transcribe");
      return;
    }
  }

  int all_frames = features.num_frames;
  int max_loop = model->max_feature_per_loop();
  int loop_count = (all_frames / max_loop) + 1;
  Token eos_id = model->eos_token_ids()[0];

  auto prompt = BuildPrompt(0);

  for (int loop_idx = 0; loop_idx < loop_count; ++loop_idx) {
    int start = loop_idx * max_loop;
    int end = std::min(start + max_loop, all_frames);
    int loop_frames = end - start;
    if (loop_frames <= 0) break;

    int n_mels = features.feature_dim;
    std::vector<float> sub_features(loop_frames * n_mels);
    for (int m = 0; m < n_mels; ++m) {
      for (int f = 0; f < loop_frames; ++f) {
        sub_features[m * loop_frames + f] =
            static_cast<float>(features.data[m * all_frames + (start + f)]);
      }
    }

    do_encode(sub_features, n_mels, loop_frames);

    reset();

    Token first_token = do_prefill(prompt);
    p.set_input_tokens(p.input_tokens() + static_cast<int>(prompt.size()));
    if (loop_idx == 0) {
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
      if (next == eos_id) break;
      if (callback) callback(next);
      prev_token = next;
    }
  }

  p.stop("transcribe");
  fill_perf_info(total_duration);
}

void Qwen3AsrContext::set_language(const std::string& language) {
  language_ = language;
}

REGISTER_MODEL(
    ASRModel, qwen3_asr, ModelSeries::kQwen3Asr,
    [](const ModelConfig& c) { return std::make_unique<Qwen3AsrModel>(c); },
    "Qwen3-ASR model");

}  // namespace houmo
