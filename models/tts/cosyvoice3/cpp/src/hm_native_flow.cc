/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_native_flow.cc
 * Description:
 *   Flow decoder implementation for CosyVoice3 TTS.
 *   Converts speech tokens to mel spectrogram using flow matching.
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

#include "hm_native_flow.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

namespace {

size_t ShapeElementCount(const std::vector<int64_t>& shape) {
  size_t count = 1;
  for (auto dim : shape) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

std::vector<houmo::TensorType> FloatToTensorVector(
    const std::vector<float>& values) {
  std::vector<houmo::TensorType> out(values.size());
  std::transform(values.begin(), values.end(), out.begin(), [](float value) {
    return static_cast<houmo::TensorType>(value);
  });
  return out;
}

std::vector<float> TensorToFloatVector(const void* data, size_t count) {
  const auto* typed = static_cast<const houmo::TensorType*>(data);
  std::vector<float> out(count);
  std::transform(
      typed, typed + count, out.begin(),
      [](houmo::TensorType value) { return static_cast<float>(value); });
  return out;
}

std::vector<float> RepeatInterleaveSeq(const std::vector<float>& input,
                                       int seq_len, int feat_dim,
                                       int repeat_times) {
  std::vector<float> out(static_cast<size_t>(seq_len * repeat_times * feat_dim),
                         0.0f);
  for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
    for (int rep_idx = 0; rep_idx < repeat_times; ++rep_idx) {
      const size_t dst_base =
          static_cast<size_t>((seq_idx * repeat_times + rep_idx) * feat_dim);
      const size_t src_base = static_cast<size_t>(seq_idx * feat_dim);
      std::copy(input.begin() + src_base, input.begin() + src_base + feat_dim,
                out.begin() + dst_base);
    }
  }
  return out;
}

std::vector<float> TransposeSeqFeatToFeatSeq(const std::vector<float>& input,
                                             int seq_len, int feat_dim) {
  std::vector<float> out(static_cast<size_t>(seq_len * feat_dim), 0.0f);
  for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx) {
    for (int feat_idx = 0; feat_idx < feat_dim; ++feat_idx) {
      out[static_cast<size_t>(feat_idx * seq_len + seq_idx)] =
          input[static_cast<size_t>(seq_idx * feat_dim + feat_idx)];
    }
  }
  return out;
}

std::vector<float> L2Normalize(const std::vector<houmo::TensorType>& input) {
  std::vector<float> out(input.size(), 0.0f);
  float squared_sum = 0.0f;
  for (size_t idx = 0; idx < input.size(); ++idx) {
    out[idx] = static_cast<float>(input[idx]);
    squared_sum += out[idx] * out[idx];
  }
  const float norm = std::sqrt(std::max(squared_sum, 1e-12f));
  for (auto& value : out) {
    value /= norm;
  }
  return out;
}

std::vector<float> BuildCosineTimeSpan(int steps) {
  constexpr float kPi = 3.14159265358979323846f;
  std::vector<float> t_span(static_cast<size_t>(steps + 1), 0.0f);
  for (int idx = 0; idx <= steps; ++idx) {
    const float t = static_cast<float>(idx) / static_cast<float>(steps);
    t_span[static_cast<size_t>(idx)] = 1.0f - std::cos(t * 0.5f * kPi);
  }
  return t_span;
}

}  // namespace

namespace houmo {

HmNativeFlow::HmNativeFlow(const std::string& input_embedding_path,
                           const std::string& pre_lookahead_layer_path,
                           const std::string& spk_embed_affine_layer_path,
                           const std::string& flow_decoder_path) {
  weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);
  auto pre_lookahead_option = tcim::Module::Option(weight_manager_);
  auto spk_embed_affine_layer_option = tcim::Module::Option(weight_manager_);
  auto flow_decoder_option = tcim::Module::Option(weight_manager_);

  pre_lookahead_layer_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(pre_lookahead_layer_->LoadModel(
      pre_lookahead_layer_path, pre_lookahead_option));

  spk_embed_affine_layer_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(spk_embed_affine_layer_->LoadModel(
      spk_embed_affine_layer_path, spk_embed_affine_layer_option));

  flow_decoder_ = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(
      flow_decoder_->LoadModel(flow_decoder_path, flow_decoder_option));

  for (int idx = 0; idx < pre_lookahead_layer_->GetInputNum(); idx++) {
    auto name = pre_lookahead_layer_->GetInputName(idx);
    auto info = pre_lookahead_layer_->GetInputInfo(name).AsContiguous();
    pre_lookahead_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
  }

  for (int idx = 0; idx < spk_embed_affine_layer_->GetInputNum(); idx++) {
    auto name = spk_embed_affine_layer_->GetInputName(idx);
    auto info = spk_embed_affine_layer_->GetInputInfo(name).AsContiguous();
    spk_embed_affine_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
  }

  for (int idx = 0; idx < flow_decoder_->GetInputNum(); idx++) {
    auto name = flow_decoder_->GetInputName(idx);
    auto info = flow_decoder_->GetInputInfo(name).AsContiguous();
    flow_decoder_input_maps_[name] = tcim::Tensor::CreateHostTensor(info);
  }

  const auto pre_lookahead_info =
      pre_lookahead_layer_->GetInputInfo(pre_lookahead_layer_->GetInputName(0));
  pre_lookahead_seq_len_ = static_cast<int>(pre_lookahead_info.Shape()[1]);
  input_embedding_dim_ = static_cast<int>(pre_lookahead_info.Shape()[2]);
  input_embedding_ = std::make_shared<HmEmbedding>(
      input_embedding_path, input_embedding_dim_, pre_lookahead_seq_len_);
}

HmNativeFlow::~HmNativeFlow() {}

std::vector<float> HmNativeFlow::Inference(
    const CosyVoice3FrontendInput& frontend_input,
    const std::vector<int>& tts_speech_tokens, CosyVoice3Perf* perf) {
  auto start_time = std::chrono::high_resolution_clock::now();
  if (tts_speech_tokens.empty()) {
    return {};
  }

  std::vector<int> flow_tokens = frontend_input.flow_prompt_speech_tokens;
  flow_tokens.insert(flow_tokens.end(), tts_speech_tokens.begin(),
                     tts_speech_tokens.end());
  const int token_len = static_cast<int>(flow_tokens.size());
  if (token_len > pre_lookahead_seq_len_) {
    throw std::runtime_error(
        "Flow token length exceeds pre-lookahead static sequence length.");
  }

  auto encoder_start_time = std::chrono::high_resolution_clock::now();
  std::vector<TensorType> token_embeddings(
      static_cast<size_t>(pre_lookahead_seq_len_ * input_embedding_dim_),
      TensorType(0));
  TensorType* embedded_ptr = input_embedding_->EmbeddingTokens(flow_tokens);
  std::copy(embedded_ptr,
            embedded_ptr +
                static_cast<size_t>(flow_tokens.size() * input_embedding_dim_),
            token_embeddings.begin());

  const auto pre_lookahead_input_name = pre_lookahead_layer_->GetInputName(0);
  CHECK_TCIM_RET_STATUS(
      pre_lookahead_input_maps_[pre_lookahead_input_name].Buffer().CopyFromHost(
          token_embeddings.data(),
          pre_lookahead_input_maps_[pre_lookahead_input_name].MemSize()));
  CHECK_TCIM_RET_STATUS(pre_lookahead_layer_->SetInput(
      pre_lookahead_input_name,
      pre_lookahead_input_maps_[pre_lookahead_input_name]));
  CHECK_TCIM_RET_STATUS(pre_lookahead_layer_->Run());
  CHECK_TCIM_RET_STATUS(pre_lookahead_layer_->Sync());

  auto pre_lookahead_output =
      pre_lookahead_layer_->GetDevOutput(pre_lookahead_layer_->GetOutputName(0))
          .ToHost(true);
  const auto pre_lookahead_output_info = pre_lookahead_layer_->GetOutputInfo(
      pre_lookahead_layer_->GetOutputName(0));
  const auto pre_lookahead_output_shape = pre_lookahead_output_info.Shape();
  const int h_seq_len = static_cast<int>(pre_lookahead_output_shape[1]);
  const int h_feat_dim = static_cast<int>(pre_lookahead_output_shape[2]);
  std::vector<float> h =
      TensorToFloatVector(pre_lookahead_output.Buffer().Data(),
                          ShapeElementCount(pre_lookahead_output_shape));
  std::vector<float> h_repeated =
      RepeatInterleaveSeq(h, h_seq_len, h_feat_dim, token_mel_ratio_);

  std::vector<float> normalized_embedding =
      L2Normalize(frontend_input.flow_embedding);
  auto spk_input_tensor = FloatToTensorVector(normalized_embedding);
  const auto spk_input_name = spk_embed_affine_layer_->GetInputName(0);
  CHECK_TCIM_RET_STATUS(
      spk_embed_affine_input_maps_[spk_input_name].Buffer().CopyFromHost(
          spk_input_tensor.data(),
          spk_embed_affine_input_maps_[spk_input_name].MemSize()));
  CHECK_TCIM_RET_STATUS(spk_embed_affine_layer_->SetInput(
      spk_input_name, spk_embed_affine_input_maps_[spk_input_name]));
  CHECK_TCIM_RET_STATUS(spk_embed_affine_layer_->Run());
  CHECK_TCIM_RET_STATUS(spk_embed_affine_layer_->Sync());

  auto spk_output =
      spk_embed_affine_layer_
          ->GetDevOutput(spk_embed_affine_layer_->GetOutputName(0))
          .ToHost(true);
  const auto spk_output_shape =
      spk_embed_affine_layer_
          ->GetOutputInfo(spk_embed_affine_layer_->GetOutputName(0))
          .Shape();
  std::vector<float> speaker_embedding = TensorToFloatVector(
      spk_output.Buffer().Data(), ShapeElementCount(spk_output_shape));
  auto encoder_end_time = std::chrono::high_resolution_clock::now();

  const int mel_len1 = frontend_input.prompt_speech_feat_len;
  const int mel_len2 = token_len * token_mel_ratio_ - mel_len1;
  if (mel_len2 <= 0) {
    return {};
  }
  const int total_mel_len = mel_len1 + mel_len2;
  const int flow_seq_len = h_seq_len * token_mel_ratio_;

  std::vector<float> conds(static_cast<size_t>(flow_seq_len * flow_mel_bins_),
                           0.0f);
  for (int idx = 0;
       idx < mel_len1 * flow_mel_bins_ &&
       idx < static_cast<int>(frontend_input.prompt_speech_feat.size());
       ++idx) {
    conds[static_cast<size_t>(idx)] = static_cast<float>(
        frontend_input.prompt_speech_feat[static_cast<size_t>(idx)]);
  }
  std::vector<float> conds_t =
      TransposeSeqFeatToFeatSeq(conds, flow_seq_len, flow_mel_bins_);
  std::vector<float> mask(static_cast<size_t>(flow_seq_len), 0.0f);
  for (int idx = 0; idx < std::min(total_mel_len, flow_seq_len); ++idx) {
    mask[static_cast<size_t>(idx)] = 1.0f;
  }
  std::vector<float> mu =
      TransposeSeqFeatToFeatSeq(h_repeated, flow_seq_len, flow_mel_bins_);

  std::mt19937 rng(std::random_device{}());
  std::normal_distribution<float> normal_dist(0.0f, 1.0f);
  std::vector<float> x(static_cast<size_t>(flow_mel_bins_ * flow_seq_len),
                       0.0f);
  for (auto& value : x) {
    value = normal_dist(rng);
  }

  const std::vector<float> t_span = BuildCosineTimeSpan(flow_steps_);
  float t = t_span[0];
  float dt = t_span[1] - t_span[0];

  std::vector<float> x_in(
      static_cast<size_t>(2 * flow_mel_bins_ * flow_seq_len), 0.0f);
  std::vector<float> mask_in(static_cast<size_t>(2 * flow_seq_len), 0.0f);
  std::vector<float> mu_in(
      static_cast<size_t>(2 * flow_mel_bins_ * flow_seq_len), 0.0f);
  std::vector<float> t_in(2, 0.0f);
  std::vector<float> cond_in(
      static_cast<size_t>(2 * flow_mel_bins_ * flow_seq_len), 0.0f);
  std::vector<float> spks_in(static_cast<size_t>(2 * flow_mel_bins_), 0.0f);

  auto decoder_start_time = std::chrono::high_resolution_clock::now();
  for (int step = 1; step < static_cast<int>(t_span.size()); ++step) {
    std::copy(x.begin(), x.end(), x_in.begin());
    std::copy(x.begin(), x.end(), x_in.begin() + x.size());
    std::copy(mask.begin(), mask.end(), mask_in.begin());
    std::copy(mask.begin(), mask.end(), mask_in.begin() + mask.size());
    std::fill(mu_in.begin(), mu_in.end(), 0.0f);
    std::copy(mu.begin(), mu.end(), mu_in.begin());
    std::fill(cond_in.begin(), cond_in.end(), 0.0f);
    std::copy(conds_t.begin(), conds_t.end(), cond_in.begin());
    std::fill(spks_in.begin(), spks_in.end(), 0.0f);
    std::copy(speaker_embedding.begin(),
              speaker_embedding.begin() +
                  std::min(speaker_embedding.size(), spks_in.size()),
              spks_in.begin());
    std::fill(t_in.begin(), t_in.end(), t);

    const auto x_tensor = FloatToTensorVector(x_in);
    const auto mask_tensor = FloatToTensorVector(mask_in);
    const auto mu_tensor = FloatToTensorVector(mu_in);
    const auto t_tensor = FloatToTensorVector(t_in);
    const auto spks_tensor = FloatToTensorVector(spks_in);
    const auto cond_tensor = FloatToTensorVector(cond_in);

    const auto x_name = flow_decoder_->GetInputName(0);
    const auto mask_name = flow_decoder_->GetInputName(1);
    const auto mu_name = flow_decoder_->GetInputName(2);
    const auto t_name = flow_decoder_->GetInputName(3);
    const auto spks_name = flow_decoder_->GetInputName(4);
    const auto cond_name = flow_decoder_->GetInputName(5);

    CHECK_TCIM_RET_STATUS(
        flow_decoder_input_maps_[x_name].Buffer().CopyFromHost(
            x_tensor.data(), flow_decoder_input_maps_[x_name].MemSize()));
    CHECK_TCIM_RET_STATUS(
        flow_decoder_input_maps_[mask_name].Buffer().CopyFromHost(
            mask_tensor.data(), flow_decoder_input_maps_[mask_name].MemSize()));
    CHECK_TCIM_RET_STATUS(
        flow_decoder_input_maps_[mu_name].Buffer().CopyFromHost(
            mu_tensor.data(), flow_decoder_input_maps_[mu_name].MemSize()));
    CHECK_TCIM_RET_STATUS(
        flow_decoder_input_maps_[t_name].Buffer().CopyFromHost(
            t_tensor.data(), flow_decoder_input_maps_[t_name].MemSize()));
    CHECK_TCIM_RET_STATUS(
        flow_decoder_input_maps_[spks_name].Buffer().CopyFromHost(
            spks_tensor.data(), flow_decoder_input_maps_[spks_name].MemSize()));
    CHECK_TCIM_RET_STATUS(
        flow_decoder_input_maps_[cond_name].Buffer().CopyFromHost(
            cond_tensor.data(), flow_decoder_input_maps_[cond_name].MemSize()));

    CHECK_TCIM_RET_STATUS(
        flow_decoder_->SetInput(x_name, flow_decoder_input_maps_[x_name]));
    CHECK_TCIM_RET_STATUS(flow_decoder_->SetInput(
        mask_name, flow_decoder_input_maps_[mask_name]));
    CHECK_TCIM_RET_STATUS(
        flow_decoder_->SetInput(mu_name, flow_decoder_input_maps_[mu_name]));
    CHECK_TCIM_RET_STATUS(
        flow_decoder_->SetInput(t_name, flow_decoder_input_maps_[t_name]));
    CHECK_TCIM_RET_STATUS(flow_decoder_->SetInput(
        spks_name, flow_decoder_input_maps_[spks_name]));
    CHECK_TCIM_RET_STATUS(flow_decoder_->SetInput(
        cond_name, flow_decoder_input_maps_[cond_name]));
    CHECK_TCIM_RET_STATUS(flow_decoder_->Run());
    CHECK_TCIM_RET_STATUS(flow_decoder_->Sync());

    auto dphi_dt_output =
        flow_decoder_->GetDevOutput(flow_decoder_->GetOutputName(0))
            .ToHost(true);
    const auto dphi_shape =
        flow_decoder_->GetOutputInfo(flow_decoder_->GetOutputName(0)).Shape();
    std::vector<float> dphi_dt_all = TensorToFloatVector(
        dphi_dt_output.Buffer().Data(), ShapeElementCount(dphi_shape));
    const size_t batch_stride =
        static_cast<size_t>(flow_mel_bins_ * flow_seq_len);
    for (size_t idx = 0; idx < batch_stride; ++idx) {
      const float dphi = dphi_dt_all[idx];
      const float cfg_dphi = dphi_dt_all[idx + batch_stride];
      x[idx] += dt * ((1.0f + inference_cfg_rate_) * dphi -
                      inference_cfg_rate_ * cfg_dphi);
    }
    t += dt;
    if (step < static_cast<int>(t_span.size()) - 1) {
      dt = t_span[static_cast<size_t>(step + 1)] - t;
    }
  }
  auto decoder_end_time = std::chrono::high_resolution_clock::now();

  std::vector<float> mel_spectrogram(
      static_cast<size_t>(flow_mel_bins_ * mel_len2), 0.0f);
  for (int mel_idx = 0; mel_idx < flow_mel_bins_; ++mel_idx) {
    const size_t src_offset =
        static_cast<size_t>(mel_idx * flow_seq_len + mel_len1);
    const size_t dst_offset = static_cast<size_t>(mel_idx * mel_len2);
    std::copy(x.begin() + src_offset, x.begin() + src_offset + mel_len2,
              mel_spectrogram.begin() + dst_offset);
  }

  if (perf) {
    auto end_time = std::chrono::high_resolution_clock::now();
    perf->flow_encoder_ms += std::chrono::duration<float, std::milli>(
                                 encoder_end_time - encoder_start_time)
                                 .count();
    perf->flow_decoder_ms += std::chrono::duration<float, std::milli>(
                                 decoder_end_time - decoder_start_time)
                                 .count();
    perf->flow_total_ms +=
        std::chrono::duration<float, std::milli>(end_time - start_time).count();
  }
  return mel_spectrogram;
}
}  // namespace houmo
