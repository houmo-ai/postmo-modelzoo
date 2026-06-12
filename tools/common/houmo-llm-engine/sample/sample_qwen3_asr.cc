/*
 * Copyright (c) 2026 HOUMO AI
 * SPDX-License-Identifier: Apache-2.0
 */

#include <filesystem>
#include <iostream>
#include <string>

#include "core/model_factory.h"
#include "models/qwen3_asr_model.h"
#include "modules/streaming_decoder.h"

int main(int argc, char* argv[]) {
  std::string audio_path, encoder_path, prefill_path, decode_path;
  std::string tokenizer_path, embedding_path;

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

  if (audio_path.empty()) {
    std::cerr << "--audio_path required\n";
    return 1;
  }

  houmo::ModelConfig config;
  config.devices = {0};
  config.prefill_path = prefill_path;
  config.decode_path = decode_path;
  config.tokenizer_path = tokenizer_path;
  config.extra_params["encoder_path"] = encoder_path;
  config.embedding_path = embedding_path;

  auto model = houmo::ModelFactory<houmo::ASRModel>::Create(
      houmo::ModelSeries::kQwen3Asr, config);

  auto ctx = model->create_context();
  auto qwen_ctx = dynamic_cast<houmo::Qwen3AsrContext*>(ctx.get());
  auto qwen_model = dynamic_cast<houmo::Qwen3AsrModel*>(model.get());
  constexpr int sample_rate = 16000;
  constexpr int stft_frames_per_second = 100;
  int encoder_window_seconds =
      qwen_model->max_feature_per_loop() / stft_frames_per_second;
  qwen_ctx->set_audio_processor(sample_rate, encoder_window_seconds,
                                encoder_window_seconds);

  houmo::StreamingDecoder decoder(qwen_model->tokenizer());
  houmo::SamplingParams params;
  qwen_ctx->Transcribe(audio_path, params,
                       [&decoder, qwen_ctx](houmo::Token token) {
                         std::string text = decoder.decode(token);
                         if (!text.empty()) {
                           // need to rfind <asr_text> to remove the prefix
                           std::cout << text;
                           std::cout.flush();
                         }
                         return true;
                       });
  std::cout << std::endl;

  std::cout << "\n";
  ctx->profiler().print_summary();
  const auto& info = qwen_ctx->perf_info();
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
