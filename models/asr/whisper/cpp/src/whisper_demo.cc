/*
 * Copyright (c) 2026 HOUMO AI
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * File: whisper_demo.cpp
 * Description: Main entry point for Whisper ASR inference demo
 */

#include <chrono>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>

#include "hm_whisper_audio.hpp"
#include "hm_whisper_infer.hpp"

#ifdef _MSC_VER
#include <Windows.h>
#endif

// Custom lightweight logger to mimic Python's loguru
namespace {

std::string GetCurrentTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                now.time_since_epoch()) %
            1000;
  auto timer = std::chrono::system_clock::to_time_t(now);
  std::tm bt = *std::localtime(&timer);
  std::ostringstream oss;
  oss << std::put_time(&bt, "%Y-%m-%d %H:%M:%S");
  oss << '.' << std::setfill('0') << std::setw(3) << ms.count();
  return oss.str();
}

#define LOGURU_LOG(level, msg) std::cout << "[" << level << "] " << msg << "\n"

#define LOG_INFO(msg) LOGURU_LOG("INFO", msg)
#define LOG_SUCCESS(msg) LOGURU_LOG("SUCCESS", msg)
#define LOG_ERROR(msg) LOGURU_LOG("ERROR", msg)

void PrintUsage(const char* program_name) {
  std::cout
      << "Whisper ASR Demo - HOUMO AI\n"
      << "==========================\n\n"
      << "Usage:\n"
      << "  " << program_name << " --audio_path <audio_file> [options]\n\n"
      << "Options:\n"
      << "  --audio_path <path>     Path to audio file (default: audio.mp3)\n"
      << "  --encode_path <path>    Path to encode model (default: "
         "output/xh2/whisper_encode.hmm)\n"
      << "  --decode_path <path>    Path to decode model (default: "
         "output/xh2/whisper_decode.hmm)\n"
      << "  --prefill_path <path>   Path to prefill model (default: "
         "output/xh2/whisper_prefill.hmm)\n"
      << "  --tokenizer_path <path> Path to tokenizer (default: "
         "whisper-medium/tokenizer.json)\n"
      << "  --chunk_size <seconds>  Audio chunk size in seconds (default: 30)\n"
      << "  --encoder_path <path>   Deprecated alias of --encode_path\n"
      << "  --decoder_path <path>   Deprecated alias of --decode_path\n"
      << "  --help, -h              Show this help message\n\n"
      << "Examples:\n"
      << "  " << program_name << " --audio_path audio.mp3\n"
      << "  " << program_name
      << " --audio_path audio.mp3 --encode_path models/whisper_encode.hmm\n\n"
      << "Environment Variables:\n"
      << "  HOUMO_TARGET     Backend target (required: xh2)\n";
}

bool CheckEnvironment() {
  const char* houmo_target_env = std::getenv("HOUMO_TARGET");
  std::string houmo_target =
      (houmo_target_env != nullptr) ? std::string(houmo_target_env) : "";

  if (houmo_target != "xh2") {
    std::cerr << "Error: Unsupported backend: " << houmo_target << "\n"
              << "Only 'xh2' backend is supported. Set HOUMO_TARGET=xh2\n";
    return false;
  }
  return true;
}

}  // namespace

int main(int argc, char* argv[]) {
#ifdef _MSC_VER
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif

  // Default paths
  std::string audio_path = "audio.mp3";
  std::string encode_path = "output/xh2/whisper_encode.hmm";
  std::string decode_path = "output/xh2/whisper_decode.hmm";
  std::string prefill_path = "output/xh2/whisper_prefill.hmm";
  std::string tokenizer_path = "whisper-medium/tokenizer.json";
  int chunk_size = 30;

  // Parse command line
  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];

    if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      return 0;
    }

    if (arg == "--audio_path" || arg == "--encode_path" ||
        arg == "--decode_path" || arg == "--encoder_path" ||
        arg == "--decoder_path" || arg == "--prefill_path" ||
        arg == "--tokenizer_path" || arg == "--chunk_size") {
      if (i + 1 >= argc) {
        std::cerr << "Error: Missing value for option: " << arg << "\n";
        PrintUsage(argv[0]);
        return 1;
      }

      std::string value = argv[++i];
      if (arg == "--audio_path") {
        audio_path = value;
      } else if (arg == "--encode_path" || arg == "--encoder_path") {
        encode_path = value;
      } else if (arg == "--decode_path" || arg == "--decoder_path") {
        decode_path = value;
      } else if (arg == "--prefill_path") {
        prefill_path = value;
      } else if (arg == "--tokenizer_path") {
        tokenizer_path = value;
      } else if (arg == "--chunk_size") {
        try {
          chunk_size = std::stoi(value);
        } catch (const std::exception&) {
          std::cerr << "Error: Invalid --chunk_size value: " << value << "\n";
          return 1;
        }

        if (chunk_size <= 0 || chunk_size > 30) {
          std::cerr << "Error: --chunk_size must be in range [1, 30]\n";
          return 1;
        }
      }
      continue;
    }

    if (!arg.empty() && arg[0] == '-') {
      std::cerr << "Error: Unknown option: " << arg << "\n";
      PrintUsage(argv[0]);
      return 1;
    }

    std::cerr << "Error: Unexpected positional argument: " << arg << "\n"
              << "Please use --audio_path <path>\n";
    PrintUsage(argv[0]);
    return 1;
  }

  if (audio_path.empty()) {
    audio_path = "audio.mp3";
  }

  // Validate input
  if (!std::filesystem::exists(audio_path)) {
    std::cerr << "Error: Audio file not found: " << audio_path << "\n";
    return 2;
  }

  // Check environment
  if (!CheckEnvironment()) {
    return 3;
  }

  const char* houmo_target_env = std::getenv("HOUMO_TARGET");
  std::string houmo_target = std::string(houmo_target_env);

  bool models_exist = std::filesystem::exists(encode_path) &&
                      std::filesystem::exists(decode_path) &&
                      std::filesystem::exists(prefill_path);

  if (!models_exist) {
    std::cerr << "Error: Model files not found. Inference skipped.\n";
    return 0;
  }

  try {
    houmo::HmWhisperInfer whisper_infer(encode_path, decode_path, prefill_path,
                                        tokenizer_path);

    LOG_INFO("encoder model loaded");
    LOG_INFO("prefill model loaded");
    LOG_INFO("decoder model loaded");

    LOG_SUCCESS("transcription:");

    // We only process single chunk info internally, but to capture performance:
    houmo::HmWhisperAudio audio_processor(chunk_size);
    audio_processor.LoadAudio(audio_path);
    int num_chunks = audio_processor.GetNumChunks();

    std::string full_transcription;
    houmo::DecodeState state;

    float total_ttft = 0.0f;
    float total_decode_cost = 0.0f;
    int total_tokens = 0;
    float total_audio_dur = 0.0f;

    // Pre-compute all mel features before inference (optimization)
    LOG_INFO("Pre-computing mel spectrograms for " +
             std::to_string(num_chunks) + " chunks...");
    auto t_preprocess_start = std::chrono::high_resolution_clock::now();
    std::vector<houmo::MelFeatures> all_mel_features;
    all_mel_features.reserve(num_chunks);
    for (int i = 0; i < num_chunks; i++) {
      all_mel_features.push_back(audio_processor.GetChunkMelFeatures(i));
    }
    auto t_preprocess_end = std::chrono::high_resolution_clock::now();
    float preprocess_time =
        std::chrono::duration<float>(t_preprocess_end - t_preprocess_start)
            .count();
    LOG_INFO("Mel pre-processing completed in " +
             std::to_string(preprocess_time) + "s");

    auto t_start_overall = std::chrono::high_resolution_clock::now();
    // Process all chunks using pre-computed mel features
    for (int i = 0; i < num_chunks; i++) {
      auto [transcription, perf] =
          whisper_infer.Transcribe(all_mel_features[i], &state);

      total_ttft += perf.ttft_time;
      total_decode_cost += perf.decode_time;
      total_tokens += perf.output_tokens;
      total_audio_dur += perf.audio_duration;

      full_transcription += transcription;
    }

    std::cout << "\n";

    auto t_end_overall = std::chrono::high_resolution_clock::now();
    float total_e2e_latency =
        std::chrono::duration<float>(t_end_overall - t_start_overall).count();

    float avg_decode_speed =
        (total_decode_cost > 0) ? (total_tokens / (total_decode_cost / 1000.0f))
                                : 0.0f;
    float avg_tpot =
        (total_tokens > 0) ? (total_decode_cost / total_tokens) : 0.0f;
    float overall_e2e_tps =
        (total_e2e_latency > 0) ? (total_tokens / total_e2e_latency) : 0.0f;
    float overall_rtf =
        (total_audio_dur > 0) ? (total_e2e_latency / total_audio_dur) : 0.0f;

    std::ostringstream oss_cost, oss_spd, oss_ttft, oss_tpot, oss_lat, oss_tps,
        oss_rtf;
    oss_cost << std::fixed << std::setprecision(3) << total_decode_cost;
    oss_spd << std::fixed << std::setprecision(2) << avg_decode_speed;
    oss_ttft << std::fixed << std::setprecision(3) << total_ttft;
    oss_tpot << std::fixed << std::setprecision(3) << avg_tpot;
    oss_lat << std::fixed << std::setprecision(3) << total_e2e_latency;
    oss_tps << std::fixed << std::setprecision(2) << overall_e2e_tps;
    oss_rtf << std::fixed << std::setprecision(4) << overall_rtf;

    LOG_SUCCESS("Output " + std::to_string(total_tokens) +
                " tokens, Decode Cost " + oss_cost.str() + " ms");
    LOG_SUCCESS("Decode Speed: " + oss_spd.str() + " tokens/s");
    LOG_SUCCESS("TTFT (Time to First Token): " + oss_ttft.str() + " ms");
    LOG_SUCCESS("TPOT (Time Per Output Token): " + oss_tpot.str() +
                " ms/token");
    LOG_SUCCESS("E2E Latency (End-to-End Latency): " + oss_lat.str() +
                " seconds");
    LOG_SUCCESS("E2E TPS (End-to-End Tokens Per Second): " + oss_tps.str() +
                " tokens/s");
    LOG_SUCCESS("RTF (Real-Time Factor): " + oss_rtf.str() +
                " (lower is better, <1 means real-time)");

  } catch (const std::exception& e) {
    std::cerr << "Error during inference: " << e.what() << "\n";
    return 5;
  }
  return 0;
}