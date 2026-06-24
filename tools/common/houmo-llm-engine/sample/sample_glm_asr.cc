/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: sample_glm_asr.cc
 * Description:
 *   GLM-ASR inference demo using Houmo framework.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SPDX-License-Identifier: Apache-2.0
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <vector>

#include "core/model_factory.h"
#include "models/glm_asr_model.h"
#include "modules/audio_processor.h"
#include "modules/streaming_decoder.h"

namespace fs = std::filesystem;

int main(int argc, char* argv[]) {
  std::string audio_path;
  std::string encoder_path;
  std::string prefill_path;
  std::string decode_path;
  std::string tokenizer_path;
  std::string embedding_path;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--audio" && i + 1 < argc)
      audio_path = argv[++i];
    else if (arg == "--encode" && i + 1 < argc)
      encoder_path = argv[++i];
    else if (arg == "--prefill" && i + 1 < argc)
      prefill_path = argv[++i];
    else if (arg == "--decode" && i + 1 < argc)
      decode_path = argv[++i];
    else if (arg == "--tokenizer" && i + 1 < argc)
      tokenizer_path = argv[++i];
    else if (arg == "--embedding" && i + 1 < argc)
      embedding_path = argv[++i];
  }

  if (audio_path.empty() || !fs::exists(audio_path)) {
    std::cerr << "Error: --audio required" << std::endl;
    return 1;
  }

  std::cout << "Loading GLM-ASR model...\n";

  houmo::ModelConfig config;
  config.devices = {0};
  config.prefill_path = prefill_path;
  config.decode_path = decode_path;
  config.tokenizer_path = tokenizer_path;
  config.extra_params["encoder_path"] = encoder_path;
  config.embedding_path = embedding_path;

  auto model = houmo::ModelFactory<houmo::ASRModel>::Create(
      houmo::ModelSeries::kGlmAsr, config);
  std::cout << "Model loaded!\n";

  auto ctx = model->create_context();
  auto glm_ctx = dynamic_cast<houmo::GlmAsrContext*>(ctx.get());
  auto glm_model = dynamic_cast<houmo::GlmAsrModel*>(model.get());
  glm_ctx->set_audio_processor(16000, 30, 30);

  houmo::StreamingDecoder decoder(glm_model->tokenizer());
  houmo::SamplingParams params;
  glm_ctx->Transcribe(audio_path, params, [&decoder](houmo::Token token) {
    std::string text = decoder.decode(token);
    if (!text.empty()) {
      std::cout << text << std::flush;
    }
    return true;
  });
  std::cout << std::endl;

  std::cout << "\n";
  ctx->profiler().print_summary();
  const auto& info = glm_ctx->perf_info();
  std::cout << "\n=== ASR RTF Metrics ===\n"
            << "Audio Duration:       " << info.audio_duration << "s\n"
            << "Chunks:               " << info.n_chunks << "\n"
            << "Output Tokens:        " << info.output_tokens << "\n"
            << "Overall RTF:          " << info.overall_rtf
            << (info.overall_rtf < 1.0f ? " (< real-time)" : "") << "\n"
            << "Inference RTF:        " << info.inference_rtf << "\n"
            << "Decode TPS:           " << info.decode_tps << " tok/s\n"
            << "Overall TPS:          " << info.overall_tps << " tok/s\n";
  return 0;
}
