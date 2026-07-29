/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Qwen3-ASR inference demo using Houmo framework.
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

#include <filesystem>
#include <iostream>
#include <string>

#include "core/model_factory.h"
#include "modules/streaming_decoder.h"
#include "qwen3_asr_model.h"

static void print_help(const char* prog) {
  std::cout << "Usage: " << prog << " [OPTIONS]\n\n"
            << "Qwen3-ASR inference demo.\n\n"
            << "Options:\n"
            << "  --audio <path>       Path to audio file (.wav, .mp3, etc.) "
               "[required]\n"
            << "  --encode <path>      Path to encoder model (.hmm)\n"
            << "  --prefill <path>     Path to prefill model (.hmm)\n"
            << "  --decode <path>      Path to decode model (.hmm)\n"
            << "  --tokenizer <path>   Path to tokenizer directory\n"
            << "  --embedding <path>   Path to embedding file (.bin or .pt)\n"
            << "  -h, --help           Show this help message\n";
}

int main(int argc, char* argv[]) {
  std::string audio_path;
  std::string encoder_path = "output/xh2/qwen3-asr-0.6b_encode.hmm";
  std::string prefill_path = "output/xh2/qwen3-asr-0.6b_prefill.hmm";
  std::string decode_path = "output/xh2/qwen3-asr-0.6b_decode.hmm";
  std::string tokenizer_path = "Qwen3-ASR-0.6B";
  std::string embedding_path = "output/xh2/hmquant/quant_embedding.bin";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      print_help(argv[0]);
      return 0;
    } else if (arg == "--audio" && i + 1 < argc)
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
    std::cerr << "Error: --audio required\n\n";
    print_help(argv[0]);
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
  params.top_k = 0;
  params.repetition_penalty = 1.0f;
  params.presence_penalty = 0.0f;
  qwen_ctx->set_sampler(params);
  qwen_ctx->Transcribe(audio_path, params,
                       [&decoder, qwen_ctx](houmo::Token token) {
                         std::string text = decoder.decode(token);
                         if (!text.empty()) {
                           // need to rfind <asr_text> to remove the prefix
                           if (text != "<asr_text>") std::cout << text;
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
