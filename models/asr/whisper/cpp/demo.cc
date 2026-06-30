/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Whisper ASR inference demo using Houmo framework.
 *
 *   Demonstrates:
 *     1. Audio loading and Mel Spectrogram extraction
 *     2. Whisper model loading (encode, prefill, decode)
 *     3. ASR transcription with streaming output
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

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "core/model_factory.h"
#include "modules/audio_processor.h"
#include "modules/streaming_decoder.h"
#include "whisper_model.h"

namespace fs = std::filesystem;

static void PrintUsage(const char* program_name) {
  std::cout
      << "Whisper ASR Demo - HOUMO AI\n"
      << "==========================\n\n"
      << "Usage:\n"
      << "  " << program_name << " --audio <audio_file> [options]\n\n"
      << "Options:\n"
      << "  --audio <path>           Path to audio file (required)\n"
      << "  --encode <path>          Path to encode model (.hmm)\n"
      << "  --prefill <path>         Path to prefill model (.hmm)\n"
      << "  --decode <path>          Path to decode model (.hmm)\n"
      << "  --tokenizer <path>       Path to tokenizer (.json)\n"
      << "  --language <code>       Language code: zh, en, auto (default: "
         "auto)\n"
      << "  --help, -h              Show this help message\n\n"
      << "Examples:\n"
      << "  " << program_name << " --audio audio.mp3\n"
      << "  " << program_name << " --audio test.wav --language zh\n\n"
      << "Environment Variables:\n"
      << "  HM_ENGINE_PATH    Base path for model files\n";
}

int main(int argc, char* argv[]) {
  std::string audio_path;
  std::string encoder_path = "output/xh2/whisper-medium_encode.hmm";
  std::string prefill_path = "output/xh2/whisper-medium_prefill.hmm";
  std::string decode_path = "output/xh2/whisper-medium_decode.hmm";
  std::string tokenizer_path = "whisper-medium";
  std::string language = "auto";

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    } else if (arg == "--audio" && i + 1 < argc) {
      audio_path = argv[++i];
    } else if (arg == "--encode" && i + 1 < argc) {
      encoder_path = argv[++i];
    } else if (arg == "--prefill" && i + 1 < argc) {
      prefill_path = argv[++i];
    } else if (arg == "--decode" && i + 1 < argc) {
      decode_path = argv[++i];
    } else if (arg == "--tokenizer" && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if (arg == "--language" && i + 1 < argc) {
      language = argv[++i];
    } else {
      std::cerr << "Unknown option: " << arg << std::endl;
      PrintUsage(argv[0]);
      return 1;
    }
  }

  if (audio_path.empty()) {
    std::cerr << "Error: --audio is required" << std::endl;
    PrintUsage(argv[0]);
    return 1;
  }

  if (!fs::exists(audio_path)) {
    std::cerr << "Error: Audio file not found: " << audio_path << std::endl;
    return 2;
  }

  if (!fs::exists(encoder_path)) {
    std::cerr << "Error: Encoder model not found: " << encoder_path
              << std::endl;
    return 3;
  }
  if (!fs::exists(prefill_path)) {
    std::cerr << "Error: Prefill model not found: " << prefill_path
              << std::endl;
    return 3;
  }
  if (!fs::exists(decode_path)) {
    std::cerr << "Error: Decode model not found: " << decode_path << std::endl;
    return 3;
  }
  if (!fs::exists(tokenizer_path)) {
    std::cerr << "Error: Tokenizer not found: " << tokenizer_path << std::endl;
    return 3;
  }

  std::cout << "========================================\n";
  std::cout << "  Whisper ASR Demo\n";
  std::cout << "========================================\n\n";

  try {
    std::cout << "Loading model...\n";

    houmo::ModelConfig config;
    config.devices = {0};
    config.prefill_path = prefill_path;
    config.decode_path = decode_path;
    config.tokenizer_path = tokenizer_path;
    config.extra_params["encoder_path"] = encoder_path;

    auto model = houmo::ModelFactory<houmo::ASRModel>::Create(
        houmo::ModelSeries::kWhisperASR, config);
    std::cout << "Model loaded successfully!\n\n";

    auto ctx = model->create_context();
    auto whisper_ctx = dynamic_cast<houmo::WhisperContext*>(ctx.get());
    auto whisper_model = dynamic_cast<houmo::WhisperModel*>(model.get());
    constexpr int sample_rate = 16000;
    constexpr int stft_frames_per_second = 100;
    int encoder_window_seconds = model->n_frames() / stft_frames_per_second;
    int chunk_seconds = encoder_window_seconds;
    whisper_ctx->set_audio_processor(sample_rate, chunk_seconds,
                                     encoder_window_seconds);

    houmo::StreamingDecoder decoder(whisper_model->tokenizer());
    houmo::SamplingParams params;
    params.top_k = 0;
    params.top_p = 1.0f;
    params.repetition_penalty = 1.1f;
    params.presence_penalty = 0.0f;
    whisper_ctx->set_language(language);
    whisper_ctx->Transcribe(audio_path, params, [&decoder](houmo::Token token) {
      std::string text = decoder.decode(token);
      if (!text.empty()) {
        std::cout << text << std::flush;
      }
      return true;
    });
    std::cout << std::endl;

    std::cout << "\n";
    ctx->profiler().print_summary();
    const auto& info = whisper_ctx->perf_info();
    std::cout << "\n=== ASR RTF Metrics ===\n"
              << "Audio Duration:       " << info.audio_duration << "s\n"
              << "Chunks:               " << info.n_chunks << "\n"
              << "Output Tokens:        " << info.output_tokens << "\n"
              << "Overall RTF:          " << info.overall_rtf
              << (info.overall_rtf < 1.0f ? " (< real-time)" : "") << "\n"
              << "Inference RTF:        " << info.inference_rtf << "\n"
              << "Decode TPS:           " << info.decode_tps << " tok/s\n"
              << "Overall TPS:          " << info.overall_tps << " tok/s\n";
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    return 5;
  }

  return 0;
}
