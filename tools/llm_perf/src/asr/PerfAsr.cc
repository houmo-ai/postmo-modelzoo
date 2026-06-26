/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: PerfAsr.cc
 * Description:
 *   ASR performance test implementation using simulated data.
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

#include "asr/PerfAsr.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <random>

#include "base/tcim_utils.h"

// ============================================================================
// PerfAsrModel
// ============================================================================

PerfAsrModel::PerfAsrModel(const houmo::ModelConfig& config)
    : ASRModel(config) {
  load();
}

PerfAsrModel::~PerfAsrModel() {
  encoder_module_.reset();
  prefill_module_.reset();
  decode_module_.reset();
}

void PerfAsrModel::load() {
  dev_manager_ = std::make_unique<tcim::DevManager>(
      tcim::DevManager::Create(config_.devices));
  weight_manager_ = std::make_unique<tcim::Module::WeightManager>(
      tcim::Module::WeightManager::CreateWeightManager(*dev_manager_));

  // Load encoder
  {
    const std::string& encoder_path = config_.extra_params.at("encoder_path");
    auto opt = tcim::Module::Option(*weight_manager_);
    opt.EnableIOLazyMode(true);
    encoder_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(encoder_module_->LoadModel(encoder_path, opt));

    auto s0 =
        encoder_module_->GetInputInfo(encoder_module_->GetInputName(0)).Shape();
    if (s0.size() >= 2) {
      n_mels_ = static_cast<int>(s0[1]);
      encoder_window_ = static_cast<int>(s0[2]);
    }

    auto input0_dtype =
        encoder_module_->GetInputInfo(encoder_module_->GetInputName(0))
            .DataType();
    encoder_input_is_f16_ = (input0_dtype == tcim::DataType::FLOAT16);

    encoder_has_input_lengths_ = (encoder_module_->GetInputNum() > 1);

    std::cout << "Encoder loaded: n_mels=" << n_mels_
              << " encoder_window=" << encoder_window_
              << " dtype=" << (encoder_input_is_f16_ ? "f16" : "f32")
              << " has_input_lengths=" << encoder_has_input_lengths_
              << std::endl;
  }

  // Load prefill
  {
    auto opt = tcim::Module::Option(*weight_manager_);
    opt.EnableIOLazyMode(true);
    prefill_module_ = std::make_shared<tcim::Module>();
    CHECK_TCIM_RET_STATUS(
        prefill_module_->LoadModel(config_.prefill_path, opt));

    auto p0 =
        prefill_module_->GetInputInfo(prefill_module_->GetInputName(0)).Shape();
    if (p0.size() >= 2) {
      max_prefill_ = static_cast<int>(p0[1]);
      hidden_size_ = static_cast<int>(p0[2]);
    }

    num_decode_layers_ = 0;
    for (int i = 0; i < prefill_module_->GetInputNum(); ++i) {
      auto name = prefill_module_->GetInputName(i);
      if (name.find("kcache") != std::string::npos) ++num_decode_layers_;
    }

    std::cout << "Prefill loaded: max_prefill=" << max_prefill_
              << " hidden_size=" << hidden_size_
              << " layers=" << num_decode_layers_ << std::endl;
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

  // KV cache sharing
  for (int i = 3; i < prefill_module_->GetInputNum(); ++i) {
    auto name = prefill_module_->GetInputName(i);
    auto dev_cache = prefill_module_->GetDevInput(name);
    decode_module_->SetDevInput(name, dev_cache);
    if (i - 2 < decode_module_->GetOutputNum()) {
      decode_module_->SetDevOutput(decode_module_->GetOutputName(i - 2),
                                   dev_cache);
    }
  }

  std::cout << "PerfAsrModel ready" << std::endl;
}

std::unique_ptr<houmo::Context> PerfAsrModel::create_context(int n_ctx) {
  if (n_ctx <= 0) n_ctx = max_prefill_;
  return std::make_unique<PerfAsrContext>(this, n_ctx);
}

// ============================================================================
// PerfAsrContext
// ============================================================================

PerfAsrContext::PerfAsrContext(houmo::ASRModel* model, int n_ctx)
    : ASRContext(model, n_ctx) {
  std::cout << "PerfAsrContext created" << std::endl;
}

// ============================================================================
// PerfRun — core entry point
// ============================================================================

AsrTranscribeResult PerfAsrContext::PerfRun(float audio_len_seconds,
                                            int token_per_second,
                                            int sample_rate) {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  int n_mels = model->n_mels_val();
  int encoder_window = model->encoder_window();
  int hidden_size = model->hidden_size();

  int n_frames = static_cast<int>(audio_len_seconds * sample_rate / 160);
  if (n_frames < 1) n_frames = 1;
  int n_chunks = (n_frames + encoder_window - 1) / encoder_window;

  profiler_.reset();
  profiler_.set_root_stage("transcribe");
  profiler_.start("transcribe");

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> mel_dist(-1.0f, 1.0f);

  int total_output_tokens = 0;

  // Pre-compute total decode rounds for progress bar
  int total_decode_rounds = 0;
  for (int ci = 0; ci < n_chunks; ++ci) {
    int lf = std::min(encoder_window, n_frames - ci * encoder_window);
    if (lf <= 0) break;
    float cal = static_cast<float>(lf) * 160 / sample_rate;
    total_decode_rounds +=
        std::max(1, static_cast<int>(cal * token_per_second));
  }

  int decode_count = 0;
  int bar_width = 50;

  for (int chunk_idx = 0; chunk_idx < n_chunks; ++chunk_idx) {
    int loop_frames =
        std::min(encoder_window, n_frames - chunk_idx * encoder_window);
    if (loop_frames <= 0) break;

    // --- Encode ---
    std::vector<float> fake_mel(n_mels * loop_frames);
    for (auto& v : fake_mel) v = mel_dist(rng);

    do_encode(fake_mel, n_mels, loop_frames);

    // --- Prefill ---
    T_out_ = static_cast<int>(audio_embeds_.size()) / hidden_size;
    do_prefill({});
    profiler_.set_input_tokens(profiler_.input_tokens() + T_out_);

    if (chunk_idx == 0) {
      profiler_.record_ttft();
    }

    // --- Decode ---
    float chunk_audio_len = static_cast<float>(loop_frames) * 160 / sample_rate;
    int chunk_decode_rounds =
        static_cast<int>(chunk_audio_len * token_per_second);
    if (chunk_decode_rounds < 1) chunk_decode_rounds = 1;

    houmo::Token prev_token = 0;
    for (int step = 0; step < chunk_decode_rounds; ++step) {
      do_decode(prev_token);
      profiler_.add_output_token();
      prev_token = 0;
      decode_count++;
      double ratio = static_cast<double>(decode_count) / total_decode_rounds;
      int filled = static_cast<int>(ratio * bar_width);
      std::cout << '\r' << "Decode: " << std::setw(3)
                << static_cast<int>(ratio * 100) << "% |"
                << std::string(filled, '*')
                << std::string(bar_width - filled, ' ') << "| " << decode_count
                << '/' << total_decode_rounds << std::flush;
    }
    total_output_tokens += chunk_decode_rounds;
  }
  std::cout << std::endl;

  profiler_.stop("transcribe");
  fill_perf_info(audio_len_seconds);

  const auto& info = perf_info();
  AsrTranscribeResult result;
  result.encode_time_ms = info.encode_time;
  result.prefill_time_ms = info.prefill_time;
  result.decode_time_ms = info.decode_time;
  result.total_time_ms = info.total_time;
  result.ttft_ms = info.ttft_time;
  result.audio_duration_s = audio_len_seconds;
  result.output_tokens = total_output_tokens;
  if (audio_len_seconds > 0) {
    result.overall_rtf =
        static_cast<float>(info.total_time / 1000.0 / audio_len_seconds);
    result.inference_rtf = static_cast<float>(
        (info.total_time - info.audio_load_time) / 1000.0 / audio_len_seconds);
  }
  if (info.decode_time > 0) {
    result.decode_tps =
        static_cast<float>(result.output_tokens) / (info.decode_time / 1000.0f);
    result.overall_tps =
        static_cast<float>(result.output_tokens) / (info.total_time / 1000.0f);
  }
  return result;
}

// ============================================================================
// Encode _impl
// ============================================================================

void PerfAsrContext::encode_preprocess_impl(const std::vector<float>& mel,
                                            int n_mels, int n_frames) {
  encode_n_frames_ = n_frames;
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  auto* enc = model->encoder_module().get();
  auto& enc_map = model->encoder_input_map();
  int encoder_window = model->encoder_window();

  std::vector<float> padded(n_mels * encoder_window, 0.0f);
  int data_frames = static_cast<int>(mel.size()) / n_mels;
  int copy_frames = std::min(n_frames, encoder_window);
  for (int m = 0; m < n_mels; ++m) {
    for (int f = 0; f < copy_frames; ++f) {
      padded[m * encoder_window + f] = mel[m * data_frames + f];
    }
  }

  std::string feat_name = enc->GetInputName(0);
  auto& feat_tensor = enc_map[feat_name];

  if (model->encoder_input_is_f16()) {
    std::vector<float16> padded_f16(padded.size());
    for (size_t i = 0; i < padded.size(); ++i)
      padded_f16[i] = static_cast<float16>(padded[i]);
    CHECK_TCIM_RET_STATUS(feat_tensor.Buffer().CopyFromHost(
        padded_f16.data(), padded_f16.size() * sizeof(float16)));
  } else {
    CHECK_TCIM_RET_STATUS(feat_tensor.Buffer().CopyFromHost(
        padded.data(), padded.size() * sizeof(float)));
  }
  enc->SetInput(feat_name, feat_tensor);

  if (model->encoder_has_input_lengths()) {
    std::string len_name = enc->GetInputName(1);
    auto& len_tensor = enc_map[len_name];
    std::vector<int> len_data = {n_frames};
    CHECK_TCIM_RET_STATUS(len_tensor.Buffer().CopyFromHost(
        len_data.data(), len_tensor.MemSize()));
    enc->SetInput(len_name, len_tensor);
  }
}

void PerfAsrContext::encode_inference_impl() {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  model->encoder_module()->Run();
  model->encoder_module()->Sync();
}

void PerfAsrContext::encode_postprocess_impl() {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  auto* enc = model->encoder_module().get();
  int hidden_size = model->hidden_size();

  std::string out_name = enc->GetOutputName(0);
  auto out_info = enc->GetOutputInfo(out_name).AsContiguous();
  tcim::Tensor out_tensor = tcim::Tensor::CreateHostTensor(out_info);
  enc->GetOutput(out_name).CastTo(out_tensor);

  size_t out_byte_size = out_tensor.MemSize();
  auto out_shape = out_info.Shape();
  int64_t out_elems = 1;
  for (auto d : out_shape) out_elems *= d;
  bool output_is_f32 =
      (out_byte_size == static_cast<size_t>(out_elems) * sizeof(float));

  int T_out = static_cast<int>(out_elems / hidden_size);
  int total = T_out * hidden_size;

  audio_embeds_.resize(total);
  if (output_is_f32) {
    const float* raw = static_cast<const float*>(out_tensor.Buffer().Data());
    for (int i = 0; i < total; ++i)
      audio_embeds_[i] = static_cast<float16>(raw[i]);
  } else {
    const float16* raw =
        static_cast<const float16*>(out_tensor.Buffer().Data());
    std::copy(raw, raw + total, audio_embeds_.begin());
  }
  T_out_ = T_out;
}

// ============================================================================
// Prefill _impl
// ============================================================================

void PerfAsrContext::prefill_preprocess_impl(
    const std::vector<houmo::Token>& /*tokens*/) {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto& prefill_map = model->prefill_input_map();
  int max_prefill = model->max_prefill();
  int hidden_size = model->hidden_size();

  std::vector<float16> fused(max_prefill * hidden_size,
                             static_cast<float16>(0.0f));
  int copy_len = std::min(T_out_, max_prefill) * hidden_size;
  std::copy(audio_embeds_.begin(), audio_embeds_.begin() + copy_len,
            fused.begin());

  prefill_seq_len_ = T_out_;

  for (int i = 0; i < prefill_module->GetInputNum(); ++i) {
    std::string name = prefill_module->GetInputName(i);
    auto& tensor = prefill_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_embeds") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(fused.data(), sz));
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

void PerfAsrContext::prefill_inference_impl() {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  model->prefill_module()->Run();
  model->prefill_module()->Sync();
}

houmo::Token PerfAsrContext::prefill_postprocess_impl() {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  auto* prefill_module = model->prefill_module().get();
  auto* decode_module = model->decode_module().get();

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
  int vocab_size =
      static_cast<int>(prefill_module->GetOutputInfo(out_name).Shape().back());

  houmo::Token first_token = static_cast<houmo::Token>(
      houmo::eigen_argmax<float16>(logits, vocab_size));

  decode_position_ = prefill_seq_len_;
  context_length_ = prefill_seq_len_;
  return first_token;
}

// ============================================================================
// Decode _impl
// ============================================================================

void PerfAsrContext::decode_preprocess_impl(houmo::Token /*prev_token*/) {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  auto* decode_module = model->decode_module().get();
  auto& decode_map = model->decode_input_map();
  int hidden_size = model->hidden_size();

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  std::vector<float16> random_embeds(hidden_size);
  for (auto& v : random_embeds) v = static_cast<float16>(dist(rng));

  for (int i = 0; i < decode_module->GetInputNum(); ++i) {
    std::string name = decode_module->GetInputName(i);
    auto& tensor = decode_map[name];
    size_t sz = tensor.MemSize();

    if (name.find("input_embeds") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(
          random_embeds.data(), hidden_size * sizeof(float16)));
    } else if (name.find("valid_length") != std::string::npos) {
      std::vector<int> data = {decode_position_};
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

void PerfAsrContext::decode_inference_impl() {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  model->decode_module()->Run();
  model->decode_module()->Sync();
}

houmo::Token PerfAsrContext::decode_postprocess_impl() {
  auto* model = static_cast<PerfAsrModel*>(asr_model());
  auto* decode_module = model->decode_module().get();

  auto out_name = decode_module->GetOutputName(0);
  auto dev_output = decode_module->GetDevOutput(out_name);
  auto host_output = dev_output.ToHost(true);
  const float16* logits =
      static_cast<const float16*>(host_output.Buffer().Data());
  int vocab_size =
      static_cast<int>(decode_module->GetOutputInfo(out_name).Shape().back());

  houmo::Token next_token = static_cast<houmo::Token>(
      houmo::eigen_argmax<float16>(logits, vocab_size));

  decode_position_++;
  context_length_++;
  return next_token;
}
