/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmAsrInfer.cc
 * Description:
 *   ASR inference wrapper implementation using libhoumo_infer.a
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SPDX-License-Identifier: Apache-2.0
 */

#include "asr/HmAsrInfer.h"

#include <chrono>
#include <iomanip>
#include <iostream>

#include "whisper_model.h"
#include "glm_asr_model.h"
#include "qwen3_asr_model.h"
#include "tcim/tcim_runtime.h"

AsrModelType HmAsrInfer::DetectModelType(const std::string& prefill_path,
                                          const std::vector<int>& devices) {
  auto dev_mgr = tcim::DevManager::Create(devices);
  auto wm = tcim::Module::WeightManager::CreateWeightManager(dev_mgr);
  auto opt = tcim::Module::Option(wm);
  opt.EnableIOLazyMode(true);

  auto mod = std::make_shared<tcim::Module>();
  CHECK_TCIM_RET_STATUS(mod->LoadModel(prefill_path, opt));

  std::string first_input = mod->GetInputName(0);
  if (first_input.find("input_ids") != std::string::npos) {
    return AsrModelType::Whisper;
  }
  if (first_input.find("input_embeds") != std::string::npos) {
    if (prefill_path.find("qwen") != std::string::npos ||
        prefill_path.find("Qwen") != std::string::npos) {
      return AsrModelType::Qwen3Asr;
    }
    return AsrModelType::GlmAsr;
  }

  throw std::runtime_error("Cannot detect ASR model type from prefill input: " +
                           first_input);
}

const char* HmAsrInfer::ModelTypeToString(AsrModelType type) {
  switch (type) {
    case AsrModelType::Whisper: return "Whisper";
    case AsrModelType::GlmAsr:  return "GLM-ASR";
    case AsrModelType::Qwen3Asr: return "Qwen3-ASR";
  }
  return "Unknown";
}

HmAsrInfer::HmAsrInfer(AsrModelType type,
                       const std::string& encode_path,
                       const std::string& prefill_path,
                       const std::string& decode_path,
                       const std::string& tokenizer_path,
                       const std::string& embedding_path,
                       const std::vector<int>& devices)
    : type_(type) {
  auto t0 = std::chrono::high_resolution_clock::now();

  houmo::ModelConfig config;
  config.devices = devices;
  config.prefill_path = prefill_path;
  config.decode_path = decode_path;
  config.tokenizer_path = tokenizer_path;
  config.embedding_path = embedding_path;
  config.extra_params["encoder_path"] = encode_path;

  houmo::ModelSeries series;
  switch (type) {
    case AsrModelType::Whisper:
      series = houmo::ModelSeries::kWhisperASR;
      break;
    case AsrModelType::GlmAsr:
      series = houmo::ModelSeries::kGlmAsr;
      break;
    case AsrModelType::Qwen3Asr:
      series = houmo::ModelSeries::kQwen3Asr;
      break;
  }

  model_ = houmo::ModelFactory<houmo::ASRModel>::Create(series, config);
  if (!model_) {
    throw std::runtime_error("Failed to create ASR model");
  }

  ctx_ = model_->create_context();

  auto t1 = std::chrono::high_resolution_clock::now();
  load_time_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

HmAsrInfer::~HmAsrInfer() {
  ctx_.reset();
  model_.reset();
}

AsrTranscribeResult HmAsrInfer::Transcribe(const std::string& audio_path) {
  AsrTranscribeResult result;
  using Clock = std::chrono::high_resolution_clock;

  auto t_total = Clock::now();

  houmo::SamplingParams params;
  params.max_tokens = 0;
  params.language = "auto";

  std::vector<houmo::Token> tokens;
  auto callback = [&tokens](houmo::Token token) {
    tokens.push_back(token);
    return true;
  };

  // Set up audio processor based on model type
  constexpr int kSampleRate = 16000;
  constexpr int kStftFramesPerSecond = 100;

  if (type_ == AsrModelType::Whisper) {
    auto* whisper_ctx = dynamic_cast<houmo::WhisperContext*>(ctx_.get());
    if (whisper_ctx) {
      int encoder_window_seconds =
          model_->n_frames() / kStftFramesPerSecond;
      whisper_ctx->set_audio_processor(kSampleRate, encoder_window_seconds,
                                        encoder_window_seconds);
    }
  } else if (type_ == AsrModelType::GlmAsr) {
    auto* glm_ctx = dynamic_cast<houmo::GlmAsrContext*>(ctx_.get());
    if (glm_ctx) {
      int encoder_window_seconds =
          model_->n_frames() / kStftFramesPerSecond;
      glm_ctx->set_audio_processor(kSampleRate, encoder_window_seconds,
                                    encoder_window_seconds);
    }
  } else if (type_ == AsrModelType::Qwen3Asr) {
    auto* qwen3_ctx = dynamic_cast<houmo::Qwen3AsrContext*>(ctx_.get());
    if (qwen3_ctx) {
      int encoder_window_seconds =
          model_->n_frames() / kStftFramesPerSecond;
      qwen3_ctx->set_audio_processor(kSampleRate, encoder_window_seconds,
                                      encoder_window_seconds);
    }
  }

  tokens.clear();
  ctx_->profiler().reset();
  ctx_->profiler().set_root_stage("transcribe");

  auto t_trans_start = Clock::now();
  auto* asr_ctx = dynamic_cast<houmo::ASRContext*>(ctx_.get());
  if (asr_ctx) {
    asr_ctx->Transcribe(audio_path, params, callback);
  } else {
    std::cerr << "Error: Context is not ASRContext" << std::endl;
    return result;
  }
  auto t_trans_end = Clock::now();

  const auto& info = asr_ctx->perf_info();

  result.output_tokens = tokens.size();
  result.audio_duration_s = info.audio_duration;
  result.audio_load_time_ms = info.audio_load_time;
  result.encode_time_ms = info.encode_time;
  result.prefill_time_ms = info.prefill_time;
  result.decode_time_ms = info.decode_time;
  result.overall_rtf = info.overall_rtf;
  result.inference_rtf = info.inference_rtf;
  result.decode_tps = info.decode_tps;
  result.overall_tps = info.overall_tps;
  result.ttft_ms = info.ttft_time;
  result.total_time_ms =
      std::chrono::duration<double, std::milli>(t_trans_end - t_trans_start)
          .count();

  ctx_->profiler().print_summary();

  double enc_chunk_s = (info.n_chunks > 0) ? result.encode_time_ms / 1000.0 / info.n_chunks : 0;
  double pref_chunk_s = (info.n_chunks > 0) ? result.prefill_time_ms / 1000.0 / info.n_chunks : 0;
  double dec_chunk_s = (info.n_chunks > 0) ? result.decode_time_ms / 1000.0 / info.n_chunks : 0;

  std::cout << "\n================================ ASR RTF Metrics ================================\n"
            << "Audio Duration:       " << result.audio_duration_s << "s\n"
            << "Chunks:               " << info.n_chunks << "\n"
            << "Output Tokens:        " << result.output_tokens << "\n"
            << "Overall RTF:          " << result.overall_rtf
            << (result.overall_rtf < 1.0f ? " (< real-time)" : "") << "\n"
            << "Inference RTF:        " << result.inference_rtf << "\n"
            << "Encode Per Chunk:     " << std::fixed << std::setprecision(2)
            << enc_chunk_s << "s\n"
            << "Prefill Per Chunk:    " << std::fixed << std::setprecision(2)
            << pref_chunk_s << "s\n"
            << "Decode Per Chunk:     " << std::fixed << std::setprecision(2)
            << dec_chunk_s << "s\n"
            << "Decode TPS:           " << std::fixed << std::setprecision(2)
            << result.decode_tps << " tok/s\n"
            << "Overall TPS:          " << std::fixed << std::setprecision(2)
            << result.overall_tps << " tok/s\n"
            << "==================================================================================\n";

  return result;
}
