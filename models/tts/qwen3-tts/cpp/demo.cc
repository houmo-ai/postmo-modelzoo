/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Qwen3-TTS CustomVoice streaming C++ inference demo.
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

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "qwen3_tts_code_predictor.h"
#include "qwen3_tts_code_predictor_embedding.h"
#include "qwen3_tts_codec_embedding.h"
#include "qwen3_tts_sampler.h"
#include "qwen3_tts_stateful_decoder.h"
#include "qwen3_tts_streaming_generator.h"
#include "qwen3_tts_streaming_prompt_builder.h"
#include "qwen3_tts_talker.h"
#include "qwen3_tts_text_embedding.h"
#include "qwen3_tts_text_processor.h"
#include "qwen3_tts_text_projection.h"
#include "qwen3_tts_wav_writer.h"

namespace {

constexpr const char* kDefaultText =
    "基于先进的存算一体技术和存储工艺，后摩智能致力于突破芯片的性能与功耗瓶颈，"
    "加速人工智能技术的普惠落地。";

constexpr houmo::Token kTtsBosTokenId = 151672;
constexpr houmo::Token kTtsEosTokenId = 151673;
constexpr houmo::Token kTtsPadTokenId = 151671;
constexpr houmo::Token kCodecThinkTokenId = 2154;
constexpr houmo::Token kCodecThinkBosTokenId = 2156;
constexpr houmo::Token kCodecThinkEosTokenId = 2157;
constexpr houmo::Token kCodecPadTokenId = 2148;
constexpr houmo::Token kCodecBosTokenId = 2149;

const std::unordered_map<std::string, houmo::Token> kLanguageTokenIds = {
    {"Chinese", 2055}, {"English", 2050}, {"Japanese", 2058},
    {"Korean", 2064},  {"French", 2061},  {"German", 2053},
    {"Spanish", 2054}, {"Italian", 2070}, {"Portuguese", 2071},
    {"Russian", 2069},
};

const std::unordered_map<std::string, houmo::Token> kSpeakerTokenIds = {
    {"vivian", 3065}, {"serena", 3066}, {"uncle_fu", 3010},
    {"ryan", 3061},   {"aiden", 2861},  {"ono_anna", 2873},
    {"sohee", 2864},  {"eric", 2875},   {"dylan", 2878},
};

const std::vector<houmo::Token> kExpectedDefaultTokens = {
    151644, 77091,  198,    104210, 102830, 24360,  69103,  100110,
    107772, 105653, 101189, 3837,   33447,  100487, 100168, 104717,
    101969, 105016, 9370,   102111, 57218,  17404,  100293, 109212,
    3837,   104293, 104455, 99361,  9370,   113734, 104368, 1773,
    151645, 198,    151644, 77091,  198,
};

std::string ModelTag(const std::string& model_name,
                     const std::string& model_size) {
  return model_name + "-" + model_size;
}

std::string DefaultHmmPath(const std::string& model_name,
                           const std::string& model_size,
                           const std::string& sub_model_name) {
  const char* target = std::getenv("HOUMO_TARGET");
  return "output/" + std::string(target == nullptr ? "xh2" : target) + "/" +
         ModelTag(model_name, model_size) + "_" + sub_model_name + ".hmm";
}

std::string DefaultHfModelDir(const std::string& model_size) {
  if (model_size == "0.6b-customvoice") {
    return "Qwen3-TTS-12Hz-0.6B-CustomVoice";
  }
  if (model_size == "1.7b-customvoice") {
    return "Qwen3-TTS-12Hz-1.7B-CustomVoice";
  }
  return "";
}

void PrintHelp(const char* program) {
  std::cout
      << "Usage: " << program << " [OPTIONS]\n\n"
      << "Run Qwen3-TTS CustomVoice streaming synthesis.\n\n"
      << "Options:\n"
      << "  --model_name <name>     Model name (default: qwen3-tts)\n"
      << "  --model_size <size>     0.6b-customvoice or 1.7b-customvoice\n"
      << "  --hf_model_dir <path>   Hugging Face model directory\n"
      << "  --text_projection_hmm <path>         TextProjection HMM file\n"
      << "  --talker_prefill_hmm <path>          Talker Prefill HMM file\n"
      << "  --talker_decode_hmm <path>           Talker Decode HMM file\n"
      << "  --talker_token_embedding <path>      Talker embedding .bin file\n"
      << "  --talker_text_embedding <path>       Text embedding .bin file\n"
      << "  --code_predictor_prefill_hmm <path>  CodePredictor Prefill HMM "
         "file\n"
      << "  --code_predictor_decode_hmm <path>   CodePredictor Decode HMM "
         "file\n"
      << "  --code_predictor_token_embedding <path> CodePredictor embedding "
         ".bin file\n"
      << "  --stateful_decoder_hmm <path>        Stateful Decoder HMM file\n"
      << "  --mode streaming                     Inference mode\n"
      << "  --chunk_size <n>                     Decoder chunk size (default: "
         "12)\n"
      << "  --text <text>                        Text to process\n"
      << "  --output_wav <path>                  Output WAV file (default: "
         "output.wav)\n"
      << "  --max-new-tokens <n>                 Maximum codec frames "
         "(default: 4096)\n"
      << "  --language <name>                    Language (default: Chinese)\n"
      << "  --speaker <name>                     Speaker (default: vivian)\n"
      << "  --seed <n>                           Sampling seed (default: "
         "1024)\n"
      << "  --device_id <n>                      Houmo device ID (default: 0)\n"
      << "  --greedy                             Disable random sampling\n"
      << "  -h, --help                           Show this help message\n";
}

bool AllFinite(const std::vector<float16>& values) {
  return std::all_of(values.begin(), values.end(), [](float16 value) {
    return std::isfinite(static_cast<float>(value));
  });
}

bool RequirePath(const std::string& option, const std::string& path) {
  if (std::filesystem::exists(path)) return true;
  std::cerr << "Error: " << option << " does not exist: " << path << "\n";
  return false;
}

class ScopedQuietOutput {
 public:
  ScopedQuietOutput() {
    old_cout_ = std::cout.rdbuf(sink_.rdbuf());
    old_cerr_ = std::cerr.rdbuf(sink_.rdbuf());
  }

  ~ScopedQuietOutput() {
    std::cout.rdbuf(old_cout_);
    std::cerr.rdbuf(old_cerr_);
  }

 private:
  std::ostringstream sink_;
  std::streambuf* old_cout_ = nullptr;
  std::streambuf* old_cerr_ = nullptr;
};

class ScopedQuietFileDescriptors {
 public:
  ScopedQuietFileDescriptors() {
    saved_stdout_ = dup(STDOUT_FILENO);
    saved_stderr_ = dup(STDERR_FILENO);
    null_fd_ = open("/dev/null", O_WRONLY);
    if (saved_stdout_ < 0 || saved_stderr_ < 0 || null_fd_ < 0 ||
        dup2(null_fd_, STDOUT_FILENO) < 0 ||
        dup2(null_fd_, STDERR_FILENO) < 0) {
      Restore();
      throw std::runtime_error("Failed to silence initialization output");
    }
  }

  ~ScopedQuietFileDescriptors() { Restore(); }

 private:
  void Restore() {
    if (restored_) return;
    if (saved_stdout_ >= 0) dup2(saved_stdout_, STDOUT_FILENO);
    if (saved_stderr_ >= 0) dup2(saved_stderr_, STDERR_FILENO);
    if (null_fd_ >= 0) close(null_fd_);
    if (saved_stdout_ >= 0) close(saved_stdout_);
    if (saved_stderr_ >= 0) close(saved_stderr_);
    restored_ = true;
  }

  bool restored_ = false;
  int saved_stdout_ = -1;
  int saved_stderr_ = -1;
  int null_fd_ = -1;
};

void LogInfo(const std::string& message) {
  std::cout << "INFO | " << message << "\n";
}

std::string Utf8Prefix(const std::string& text, size_t character_count) {
  size_t offset = 0;
  size_t characters = 0;
  while (offset < text.size() && characters < character_count) {
    const unsigned char byte = static_cast<unsigned char>(text[offset]);
    size_t width = 1;
    if ((byte & 0xE0) == 0xC0) {
      width = 2;
    } else if ((byte & 0xF0) == 0xE0) {
      width = 3;
    } else if ((byte & 0xF8) == 0xF0) {
      width = 4;
    }
    if (offset + width > text.size()) break;
    offset += width;
    ++characters;
  }
  return text.substr(0, offset);
}

void PrintPerfRow(const std::string& component, double seconds,
                  double total_seconds, size_t count,
                  const std::string& description) {
  const double percentage =
      total_seconds > 0.0 ? seconds / total_seconds * 100.0 : 0.0;
  const double average_ms = count > 0 ? seconds * 1000.0 / count : 0.0;
  std::cout << "| " << std::left << std::setw(20) << component << " | "
            << std::right << std::setw(7) << std::fixed << std::setprecision(2)
            << seconds << " | " << std::setw(5) << std::setprecision(1)
            << percentage << "% | ";
  if (count > 0) {
    std::cout << std::setw(5) << count << " | " << std::setw(7)
              << std::setprecision(2) << average_ms;
  } else {
    std::cout << "      |        ";
  }
  std::cout << " | " << std::left << std::setw(46) << description << " |\n";
}

void PrintPerformanceBreakdown(const houmo::Qwen3TTSGenerationPerf& perf,
                               double embedding_seconds, double decoder_seconds,
                               size_t decoder_count, double total_seconds) {
  const double talker_seconds = perf.talker_prefill_seconds +
                                perf.talker_decode_seconds +
                                perf.talker_sampling_seconds;
  const double predictor_seconds =
      perf.predictor_prepare_seconds + perf.predictor_prefill_seconds +
      perf.predictor_decode_seconds + perf.predictor_sampling_seconds;
  const double measured_seconds = embedding_seconds + talker_seconds +
                                  perf.frame_prepare_seconds +
                                  predictor_seconds + decoder_seconds;
  const double other_seconds = std::max(0.0, total_seconds - measured_seconds);

  std::cout << "+--------------------------------------------------------------"
               "--------------------------------------------+\n"
            << "|                                     Streaming Performance "
               "Breakdown:                                     |\n"
            << "+----------------------+---------+--------+-------+---------+--"
               "----------------------------------------------+\n"
            << "| Component            | Time(s) |    Pct | Count | Avg(ms) | "
               "Description                                    |\n"
            << "+----------------------+---------+--------+-------+---------+--"
               "----------------------------------------------+\n";
  PrintPerfRow("  embedding_prep", embedding_seconds, total_seconds, 0,
               "Prepare model input embeddings");
  PrintPerfRow("  talker", talker_seconds, total_seconds, 0,
               "All Talker model computation");
  PrintPerfRow("    prefill", perf.talker_prefill_seconds, total_seconds,
               perf.talker_prefill_count,
               "Run the initial Talker forward pass");
  PrintPerfRow("    decode", perf.talker_decode_seconds, total_seconds,
               perf.talker_decode_count,
               "Generate codec-group-0 tokens autoregressively");
  PrintPerfRow("    sampling", perf.talker_sampling_seconds, total_seconds,
               perf.talker_sampling_count, "Sample tokens from Talker logits");
  PrintPerfRow("  frame_prepare", perf.frame_prepare_seconds, total_seconds, 0,
               "Prepare per-frame generation inputs");
  PrintPerfRow("  code_predictor", predictor_seconds, total_seconds, 0,
               "All Code Predictor computation");
  PrintPerfRow("    prepare", perf.predictor_prepare_seconds, total_seconds, 0,
               "Prepare Code Predictor inputs");
  PrintPerfRow("    prefill", perf.predictor_prefill_seconds, total_seconds,
               perf.predictor_prefill_count, "Run Code Predictor prefill");
  PrintPerfRow("    decode", perf.predictor_decode_seconds, total_seconds,
               perf.predictor_decode_count, "Generate remaining codec groups");
  PrintPerfRow("    sampling", perf.predictor_sampling_seconds, total_seconds,
               perf.predictor_sampling_count,
               "Sample tokens from Code Predictor logits");
  PrintPerfRow("  stateful_decoder", decoder_seconds, total_seconds,
               decoder_count, "Decode generated codec chunks into waveforms");
  PrintPerfRow("  other", other_seconds, total_seconds, 0,
               "Time outside individually tracked metrics");
  PrintPerfRow("  total", total_seconds, total_seconds, 0,
               "End-to-end inference time");
  std::cout << "+----------------------+---------+--------+-------+---------+--"
               "----------------------------------------------+\n";
}

struct DemoOptions {
  std::string text = kDefaultText;
  std::string tokenizer_path;
  std::string text_embedding_path = "output/xh2/hmquant/text_embedding.bin";
  std::string codec_embedding_path = "output/xh2/hmquant/quant_embedding.bin";
  std::string text_projection_path;
  std::string talker_prefill_path;
  std::string talker_decode_path;
  std::string predictor_prefill_path;
  std::string predictor_decode_path;
  std::string predictor_embedding_path =
      "output/xh2/hmquant/quant_embedding_code_predictor.bin";
  std::string stateful_decoder_path;
  std::string output_path = "output.wav";
  size_t max_frames = 4096;
  bool do_sample = true;
  std::string language = "Chinese";
  std::string speaker = "vivian";
  std::string model_name = "qwen3-tts";
  std::string model_size = "0.6b-customvoice";
  std::string mode = "streaming";
  uint32_t seed = 1024;
  int device_id = 0;
  size_t chunk_size = houmo::Qwen3TTSStatefulDecoder::kChunkSize;
};

using Clock = std::chrono::steady_clock;

class AudioCollector {
 public:
  explicit AudioCollector(Clock::time_point start) : start_(start) {}

  void Emit(std::vector<float> audio) {
    if (audio.empty()) return;
    const auto emit_time = Clock::now();
    ++chunk_count_;
    LogFirstChunk(emit_time);
    const double audio_ms =
        static_cast<double>(audio.size()) / 24000.0 * 1000.0;
    const std::optional<double> gap_ms = MeasureGap(emit_time);
    previous_chunk_time_ = emit_time;
    previous_chunk_audio_ms_ = audio_ms;
    LogChunk(audio.size(), audio_ms, gap_ms);
    audio_.insert(audio_.end(), audio.begin(), audio.end());
  }

  const std::vector<float>& audio() const { return audio_; }
  const std::optional<Clock::time_point>& first_audio_time() const {
    return first_audio_time_;
  }
  size_t chunk_count() const { return chunk_count_; }
  size_t playback_gap_chunks() const { return playback_gap_chunks_; }
  double max_playback_gap_ms() const { return max_playback_gap_ms_; }
  double total_playback_gap_ms() const { return total_playback_gap_ms_; }

 private:
  void LogFirstChunk(Clock::time_point emit_time) {
    if (first_audio_time_.has_value()) return;
    first_audio_time_ = emit_time;
    const double latency_ms =
        std::chrono::duration<double, std::milli>(emit_time - start_).count();
    std::ostringstream message;
    message << std::fixed << std::setprecision(1)
            << "First audio chunk latency: " << latency_ms << "ms";
    LogInfo(message.str());
  }

  std::optional<double> MeasureGap(Clock::time_point emit_time) {
    if (!previous_chunk_time_.has_value()) return std::nullopt;
    const double gap_ms = std::max(
        0.0, std::chrono::duration<double, std::milli>(
                 emit_time - *previous_chunk_time_)
                     .count() -
                 previous_chunk_audio_ms_);
    if (gap_ms > 0.0) {
      ++playback_gap_chunks_;
      total_playback_gap_ms_ += gap_ms;
      max_playback_gap_ms_ = std::max(max_playback_gap_ms_, gap_ms);
    }
    return gap_ms;
  }

  void LogChunk(size_t sample_count, double audio_ms,
                const std::optional<double>& gap_ms) const {
    std::ostringstream message;
    message << std::fixed << std::setprecision(0) << "  Chunk " << chunk_count_
            << ": " << sample_count << " samples (" << audio_ms << "ms audio)";
    if (gap_ms.has_value()) {
      message << std::setprecision(1) << " | playback_gap: " << *gap_ms << "ms";
    }
    LogInfo(message.str());
  }

  Clock::time_point start_;
  std::optional<Clock::time_point> first_audio_time_;
  std::optional<Clock::time_point> previous_chunk_time_;
  double previous_chunk_audio_ms_ = 0.0;
  size_t chunk_count_ = 0;
  size_t playback_gap_chunks_ = 0;
  double max_playback_gap_ms_ = 0.0;
  double total_playback_gap_ms_ = 0.0;
  std::vector<float> audio_;
};

houmo::Qwen3TTSSamplingConfig CreateTalkerSamplingConfig(bool do_sample,
                                                         uint32_t seed) {
  houmo::Qwen3TTSSamplingConfig config;
  config.do_sample = do_sample;
  config.seed = seed;
  config.temperature = 0.9f;
  config.top_k = 50;
  config.repetition_penalty = 1.05f;
  config.min_new_tokens = 2;
  config.eos_token_id = 2150;
  for (houmo::Token token = 2048; token < 3072; ++token) {
    if (token != config.eos_token_id) config.suppress_tokens.push_back(token);
  }
  return config;
}

houmo::Qwen3TTSSamplingConfig CreatePredictorSamplingConfig(bool do_sample,
                                                            uint32_t seed) {
  houmo::Qwen3TTSSamplingConfig config;
  config.do_sample = do_sample;
  config.seed = seed;
  config.temperature = 0.9f;
  config.top_k = 50;
  return config;
}

std::optional<bool> ParseStringOption(const std::string& arg, int argc,
                                      char* argv[], int* index,
                                      DemoOptions* options) {
  const std::unordered_map<std::string, std::string*> string_options = {
      {"--model_name", &options->model_name},
      {"--model_size", &options->model_size},
      {"--hf_model_dir", &options->tokenizer_path},
      {"--tokenizer", &options->tokenizer_path},
      {"--mode", &options->mode},
      {"--text", &options->text},
      {"--text-embedding", &options->text_embedding_path},
      {"--talker_text_embedding", &options->text_embedding_path},
      {"--codec-embedding", &options->codec_embedding_path},
      {"--talker_token_embedding", &options->codec_embedding_path},
      {"--text-projection", &options->text_projection_path},
      {"--text_projection_hmm", &options->text_projection_path},
      {"--talker_prefill_hmm", &options->talker_prefill_path},
      {"--talker_decode_hmm", &options->talker_decode_path},
      {"--code_predictor_prefill_hmm", &options->predictor_prefill_path},
      {"--code_predictor_decode_hmm", &options->predictor_decode_path},
      {"--code_predictor_token_embedding", &options->predictor_embedding_path},
      {"--stateful_decoder_hmm", &options->stateful_decoder_path},
      {"--output", &options->output_path},
      {"--output_wav", &options->output_path},
      {"--language", &options->language},
      {"--speaker", &options->speaker},
  };
  const auto option = string_options.find(arg);
  if (option == string_options.end()) return std::nullopt;
  if (*index + 1 >= argc) {
    std::cerr << "Error: incomplete option: " << arg << "\n";
    return false;
  }
  *(option->second) = argv[++*index];
  return true;
}

std::optional<bool> ParseNumericOption(const std::string& arg, int argc,
                                       char* argv[], int* index,
                                       DemoOptions* options) {
  if (arg != "--chunk_size" && arg != "--max-frames" &&
      arg != "--max-new-tokens" && arg != "--seed" &&
      arg != "--device_id") {
    return std::nullopt;
  }
  if (*index + 1 >= argc) {
    std::cerr << "Error: incomplete option: " << arg << "\n";
    return false;
  }
  const std::string value = argv[++*index];
  try {
    size_t parsed_chars = 0;
    if (arg != "--device_id" && !value.empty() && value.front() == '-') {
      throw std::invalid_argument("negative unsigned value");
    }
    if (arg == "--chunk_size") {
      options->chunk_size = std::stoul(value, &parsed_chars);
    } else if (arg == "--max-frames" || arg == "--max-new-tokens") {
      options->max_frames = std::stoul(value, &parsed_chars);
    } else if (arg == "--seed") {
      const unsigned long seed = std::stoul(value, &parsed_chars);
      if (seed > std::numeric_limits<uint32_t>::max()) {
        throw std::out_of_range("seed");
      }
      options->seed = static_cast<uint32_t>(seed);
    } else {
      options->device_id = std::stoi(value, &parsed_chars);
    }
    if (parsed_chars != value.size()) {
      std::cerr << "Error: invalid value for " << arg << ": " << value << "\n";
      return false;
    }
  } catch (const std::exception&) {
    std::cerr << "Error: invalid value for " << arg << ": " << value << "\n";
    return false;
  }
  return true;
}

enum class ParseResult { kSuccess, kHelp, kError };

ParseResult ParseArgument(const std::string& arg, int argc, char* argv[],
                          int* index, DemoOptions* options) {
  if (arg == "--help" || arg == "-h") {
    PrintHelp(argv[0]);
    return ParseResult::kHelp;
  }

  const auto string_result = ParseStringOption(arg, argc, argv, index, options);
  if (string_result.has_value()) {
    return *string_result ? ParseResult::kSuccess : ParseResult::kError;
  }

  const auto numeric_result =
      ParseNumericOption(arg, argc, argv, index, options);
  if (numeric_result.has_value()) {
    return *numeric_result ? ParseResult::kSuccess : ParseResult::kError;
  }

  if (arg == "--greedy") {
    options->do_sample = false;
    return ParseResult::kSuccess;
  }

  std::cerr << "Error: unknown or incomplete option: " << arg << "\n\n";
  PrintHelp(argv[0]);
  return ParseResult::kError;
}

ParseResult ParseOptions(int argc, char* argv[], DemoOptions* options) {
  for (int index = 1; index < argc; ++index) {
    const ParseResult result =
        ParseArgument(argv[index], argc, argv, &index, options);
    if (result != ParseResult::kSuccess) return result;
  }
  return ParseResult::kSuccess;
}

bool ResolveOptions(DemoOptions* options) {
  if (options->model_name != "qwen3-tts") {
    std::cerr << "Error: unsupported model_name: " << options->model_name
              << "\n";
    return false;
  }
  if (options->model_size != "0.6b-customvoice" &&
      options->model_size != "1.7b-customvoice") {
    std::cerr << "Error: unsupported model_size: " << options->model_size
              << "; supported values: 0.6b-customvoice, 1.7b-customvoice\n";
    return false;
  }
  if (options->mode != "streaming") {
    std::cerr << "Error: only streaming mode is supported by C++ demo\n";
    return false;
  }
  if (options->chunk_size != houmo::Qwen3TTSStatefulDecoder::kChunkSize) {
    std::cerr << "Error: chunk_size must be "
              << houmo::Qwen3TTSStatefulDecoder::kChunkSize << "\n";
    return false;
  }
  if (options->tokenizer_path.empty()) {
    options->tokenizer_path = DefaultHfModelDir(options->model_size);
  }
  if (options->tokenizer_path.empty()) {
    std::cerr << "Error: unsupported model_size: " << options->model_size
              << "\n";
    return false;
  }
  if (options->text_projection_path.empty()) {
    options->text_projection_path =
        DefaultHmmPath(options->model_name, options->model_size,
                       "text_projection");
  }
  if (options->talker_prefill_path.empty()) {
    options->talker_prefill_path =
        DefaultHmmPath(options->model_name, options->model_size, "talker_prefill");
  }
  if (options->talker_decode_path.empty()) {
    options->talker_decode_path =
        DefaultHmmPath(options->model_name, options->model_size, "talker_decode");
  }
  if (options->predictor_prefill_path.empty()) {
    options->predictor_prefill_path = DefaultHmmPath(
        options->model_name, options->model_size, "code_predictor_prefill");
  }
  if (options->predictor_decode_path.empty()) {
    options->predictor_decode_path = DefaultHmmPath(
        options->model_name, options->model_size, "code_predictor_decode");
  }
  if (options->stateful_decoder_path.empty()) {
    options->stateful_decoder_path =
        DefaultHmmPath(options->model_name, options->model_size,
                       "stateful_decoder");
  }
  return !options->tokenizer_path.empty() &&
         RequirePath("--tokenizer", options->tokenizer_path) &&
         RequirePath("--text-embedding", options->text_embedding_path) &&
         RequirePath("--codec-embedding", options->codec_embedding_path) &&
         RequirePath("--text-projection", options->text_projection_path) &&
         RequirePath("talker prefill", options->talker_prefill_path) &&
         RequirePath("talker decode", options->talker_decode_path) &&
         RequirePath("CodePredictor prefill", options->predictor_prefill_path) &&
         RequirePath("CodePredictor decode", options->predictor_decode_path) &&
         RequirePath("CodePredictor embedding", options->predictor_embedding_path) &&
         RequirePath("stateful decoder", options->stateful_decoder_path);
}

int RunDemo(const DemoOptions& options) {
  const auto& text = options.text;
  const auto& tokenizer_path = options.tokenizer_path;
  const auto& text_embedding_path = options.text_embedding_path;
  const auto& codec_embedding_path = options.codec_embedding_path;
  const auto& text_projection_path = options.text_projection_path;
  const auto& talker_prefill_path = options.talker_prefill_path;
  const auto& talker_decode_path = options.talker_decode_path;
  const auto& predictor_prefill_path = options.predictor_prefill_path;
  const auto& predictor_decode_path = options.predictor_decode_path;
  const auto& predictor_embedding_path = options.predictor_embedding_path;
  const auto& stateful_decoder_path = options.stateful_decoder_path;
  const auto& output_path = options.output_path;
  const size_t max_frames = options.max_frames;
  const bool do_sample = options.do_sample;
  const auto& language = options.language;
  const auto& speaker = options.speaker;
  const auto& model_name = options.model_name;
  const auto& model_size = options.model_size;
  const uint32_t seed = options.seed;
  const int device_id = options.device_id;

  try {
    if (device_id < 0) {
      throw std::invalid_argument("device_id must be non-negative");
    }
    const auto language_it = kLanguageTokenIds.find(language);
    if (language_it == kLanguageTokenIds.end()) {
      throw std::invalid_argument("Unsupported language: " + language);
    }
    const auto speaker_it = kSpeakerTokenIds.find(speaker);
    if (speaker_it == kSpeakerTokenIds.end()) {
      throw std::invalid_argument("Unsupported speaker: " + speaker);
    }
    std::unique_ptr<houmo::Qwen3TTSTextProcessor> processor;
    std::unique_ptr<houmo::Qwen3TTSTextEmbedding> text_embedding;
    std::unique_ptr<houmo::Qwen3TTSTextProjection> text_projection;
    std::unique_ptr<houmo::Qwen3TTSCodecEmbedding> codec_embedding;
    std::unique_ptr<houmo::Qwen3TTSTalker> talker;
    std::unique_ptr<houmo::Qwen3TTSCodePredictor> predictor;
    std::unique_ptr<houmo::Qwen3TTSCodePredictorEmbedding> predictor_embedding;
    std::unique_ptr<houmo::Qwen3TTSStatefulDecoder> decoder;
    {
      ScopedQuietOutput quiet;
      ScopedQuietFileDescriptors quiet_fds;
      processor =
          std::make_unique<houmo::Qwen3TTSTextProcessor>(tokenizer_path);
      text_embedding =
          std::make_unique<houmo::Qwen3TTSTextEmbedding>(text_embedding_path);
      text_projection = std::make_unique<houmo::Qwen3TTSTextProjection>(
          text_projection_path, device_id);
      talker = std::make_unique<houmo::Qwen3TTSTalker>(
          talker_prefill_path, talker_decode_path, device_id);
      codec_embedding = std::make_unique<houmo::Qwen3TTSCodecEmbedding>(
          codec_embedding_path, static_cast<int>(talker->hidden_dim()));
      predictor = std::make_unique<houmo::Qwen3TTSCodePredictor>(
          predictor_prefill_path, predictor_decode_path, device_id);
      predictor_embedding =
          std::make_unique<houmo::Qwen3TTSCodePredictorEmbedding>(
              predictor_embedding_path, predictor->hidden_dim());
      decoder = std::make_unique<houmo::Qwen3TTSStatefulDecoder>(
          stateful_decoder_path, device_id);
    }

    const auto talker_sampling =
        CreateTalkerSamplingConfig(do_sample, seed);
    const auto predictor_sampling =
        CreatePredictorSamplingConfig(do_sample, seed);

    auto sampling_random = std::make_shared<std::mt19937>(seed);
    houmo::Qwen3TTSStreamingGenerator generator(
        talker.get(), codec_embedding.get(), predictor.get(),
        predictor_embedding.get(),
        houmo::Qwen3TTSSampler(talker_sampling, sampling_random),
        houmo::Qwen3TTSSampler(predictor_sampling, sampling_random));
    houmo::Qwen3TTSWavWriter wav_writer(output_path);
    std::vector<houmo::Qwen3TTSCodecFrame> decoder_frames;
    decoder_frames.reserve(houmo::Qwen3TTSStatefulDecoder::kChunkSize);

    LogInfo("Running Qwen3-TTS streaming demo: model_name=" + model_name +
            ", model_size=" + model_size + ", hf_model_dir=" + tokenizer_path);
    const auto e2e_start = Clock::now();
    LogInfo("Starting streaming TTS for text: " + Utf8Prefix(text, 50) + "...");
    auto decoder_state = decoder->CreateState();
    const auto ttfa_start = Clock::now();
    double embedding_seconds = 0.0;
    double decoder_seconds = 0.0;
    size_t decoder_count = 0;
    AudioCollector audio_collector(ttfa_start);

    const auto embedding_start = Clock::now();
    const auto features = processor->Process(text);
    const auto role_embedding = text_embedding->Lookup(features.role_ids);
    const auto body_embedding = text_embedding->Lookup(features.body_ids);
    const auto role_hidden = text_projection->Project(role_embedding);
    const auto body_hidden = text_projection->Project(body_embedding);
    const auto special_hidden = text_projection->Project(text_embedding->Lookup(
        {kTtsBosTokenId, kTtsEosTokenId, kTtsPadTokenId}));
    const size_t hidden_dim = special_hidden.hidden_dim;
    const auto special_token = [&](size_t index) {
      houmo::Qwen3TTSHiddenSequence token;
      token.sequence_length = 1;
      token.hidden_dim = hidden_dim;
      const auto begin = special_hidden.data.begin() +
                         static_cast<std::ptrdiff_t>(index * hidden_dim);
      token.data.assign(begin, begin + static_cast<std::ptrdiff_t>(hidden_dim));
      return token;
    };
    const auto codec_prompt_hidden = codec_embedding->Lookup(
        {kCodecThinkTokenId, kCodecThinkBosTokenId, language_it->second,
         kCodecThinkEosTokenId, speaker_it->second, kCodecPadTokenId,
         kCodecBosTokenId});
    houmo::Qwen3TTSStreamingPromptBuilder prompt_builder;
    const auto prompt = prompt_builder.Build(
        role_hidden, body_hidden, special_token(0), special_token(1),
        special_token(2), codec_prompt_hidden);
    embedding_seconds =
        std::chrono::duration<double>(Clock::now() - embedding_start).count();

    const size_t generated_frames = generator.Generate(
        prompt, max_frames, [&](const houmo::Qwen3TTSCodecFrame& frame) {
          decoder_frames.push_back(frame);
          if (decoder_frames.size() ==
              houmo::Qwen3TTSStatefulDecoder::kChunkSize) {
            const auto decoder_start = Clock::now();
            auto decoded = decoder->Decode(decoder_frames,
                                           std::move(decoder_state), false);
            decoder_seconds +=
                std::chrono::duration<double>(Clock::now() - decoder_start)
                    .count();
            ++decoder_count;
            audio_collector.Emit(std::move(decoded.audio));
            decoder_state = std::move(decoded.state);
            decoder_frames.clear();
          }
          return true;
        });
    const auto decoder_start = Clock::now();
    auto decoded =
        decoder->Decode(decoder_frames, std::move(decoder_state), true);
    decoder_seconds +=
        std::chrono::duration<double>(Clock::now() - decoder_start).count();
    ++decoder_count;
    audio_collector.Emit(std::move(decoded.audio));
    decoder_state = std::move(decoded.state);
    const auto inference_end = Clock::now();
    wav_writer.Write(audio_collector.audio());
    wav_writer.Close();

    const double e2e_seconds =
        std::chrono::duration<double>(inference_end - e2e_start).count();
    const double audio_seconds =
        static_cast<double>(wav_writer.sample_count()) / 24000.0;
    const double rtf = audio_seconds > 0.0 ? e2e_seconds / audio_seconds : 0.0;
    const double ttfa_ms = audio_collector.first_audio_time().has_value()
                               ? std::chrono::duration<double, std::milli>(
                                     *audio_collector.first_audio_time() -
                                     ttfa_start)
                                     .count()
                               : 0.0;

    const auto& generation_perf = generator.perf();
    if (generation_perf.reached_eos) {
      LogInfo("Reached EOS token at step " +
              std::to_string(generation_perf.eos_step));
    } else if (generation_perf.reached_max_frames) {
      LogInfo("Reached max_new_tokens at step " +
              std::to_string(generated_frames));
    }
    std::ostringstream summary;
    summary << std::fixed << std::setprecision(2) << "Audio saved to "
            << output_path << " | duration: " << audio_seconds
            << "s | inference: " << e2e_seconds
            << "s | RTF: " << std::setprecision(4) << rtf
            << " | chunks: " << audio_collector.chunk_count()
            << " | first_chunk_latency: " << std::setprecision(1) << ttfa_ms
            << "ms | playback_gap_chunks: "
            << audio_collector.playback_gap_chunks()
            << " | max_playback_gap: " << audio_collector.max_playback_gap_ms()
            << "ms | total_playback_gap: "
            << audio_collector.total_playback_gap_ms() << "ms";
    LogInfo(summary.str());
    PrintPerformanceBreakdown(generation_perf, embedding_seconds,
                              decoder_seconds, decoder_count, e2e_seconds);

    const bool sizes_valid =
        prompt.initial_prompt.sequence_length == 10 &&
        prompt.initial_prompt.hidden_dim == talker->hidden_dim() &&
        prompt.trailing_text_hidden.sequence_length == features.body_ids.size();
    if (!sizes_valid || !AllFinite(prompt.initial_prompt.data) ||
        !AllFinite(prompt.trailing_text_hidden.data)) {
      std::cerr << "Error: streaming prompt output is invalid\n";
      return 2;
    }

    if (text == kDefaultText && features.input_ids != kExpectedDefaultTokens) {
      std::cerr
          << "Error: default text tokens differ from the Python reference\n";
      return 3;
    }

  } catch (const std::exception& error) {
    std::cerr << "Error: " << error.what() << "\n";
    return 1;
  }
  return 0;
}

}  // namespace

int main(int argc, char* argv[]) {
  DemoOptions options;
  const ParseResult parse_result = ParseOptions(argc, argv, &options);
  if (parse_result == ParseResult::kHelp) return 0;
  if (parse_result == ParseResult::kError || !ResolveOptions(&options)) {
    return 1;
  }
  return RunDemo(options);
}
