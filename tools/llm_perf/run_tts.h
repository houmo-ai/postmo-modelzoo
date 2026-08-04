/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: run_tts.h
 * Description:
 *   TTS performance test runner using fixed-frame simulated text input.
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

#ifndef RUN_TTS_H
#define RUN_TTS_H

#include <algorithm>
#include <charconv>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <system_error>
#include <unordered_map>
#include <utility>
#include <vector>

#include "qwen3_tts_wav_writer.h"
#include "tts/PerfTts.h"
#include "utils/device_monitor/device_monitor.h"
#include "utils/perf_dumper/perf_dumper.h"
#include "utils/utils.h"
#if defined(__linux__)
#include "utils/host_monitor/host_monitor.h"
#endif

static constexpr float kTtsAlarmTemperatureThreshold = 80.0f;
static constexpr float kTtsShutdownTemperatureThreshold = 100.0f;

static int ValidateTtsPositiveInteger(
    const std::unordered_map<std::string, std::string>& args,
    const std::string& name) {
  const auto it = args.find(name);
  if (it == args.end() || it->second.empty()) {
    throw std::invalid_argument("Missing or empty TTS argument: " + name);
  }

  int value = 0;
  const char* begin = it->second.data();
  const char* end = begin + it->second.size();
  const auto parsed = std::from_chars(begin, end, value);
  if (parsed.ec != std::errc() || parsed.ptr != end || value <= 0) {
    throw std::invalid_argument(name + " must be a positive integer");
  }
  return value;
}

static void RejectUnsupportedTtsArgs(
    const std::unordered_map<std::string, std::string>& args) {
  static const std::set<std::string> unsupported = {
      "input",      "output",         "mode",           "seed",
      "text",       "tokenizer",      "language",       "speaker",
      "chunk_size", "max_new_tokens", "max-new-tokens", "max_frames",
      "greedy",     "temperature",    "top_k",          "top_p",
      "chunk",      "batch",          "prefill",        "decode",
      "visual",     "encode",         "device_id",      "LazyMode",
      "skip_perf",  "warm_up_input",  "warm_up_output"};
  for (const auto& name : unsupported) {
    if (args.count(name)) {
      throw std::invalid_argument("Argument '" + name +
                                  "' is not supported for TTS perf");
    }
  }
}

static TtsPerfSettings ParseTtsSettings(
    std::unordered_map<std::string, std::string> args) {
  RejectUnsupportedTtsArgs(args);

  TtsPerfSettings settings;
  settings.requested_audio_length_s =
      validate_positive_double(args, "tts_audio_length");
  settings.token_per_second =
      args.count("token_per_second")
          ? ValidateTtsPositiveInteger(args, "token_per_second")
          : 3;
  settings.text_projection_path =
      validate_path(args, "text_projection").string();
  settings.talker_prefill_path = validate_path(args, "talker_prefill").string();
  settings.talker_decode_path = validate_path(args, "talker_decode").string();
  settings.code_predictor_prefill_path =
      validate_path(args, "code_predictor_prefill").string();
  settings.code_predictor_decode_path =
      validate_path(args, "code_predictor_decode").string();
  settings.stateful_decoder_path =
      validate_path(args, "stateful_decoder").string();
  settings.embedding_path = validate_path(args, "embedding").string();
  settings.code_embedding_path = validate_path(args, "code_embedding").string();
  settings.text_embedding_path = validate_path(args, "text_embedding").string();

  std::vector<int> devices;
  if (args.count("devices")) {
    devices = validate_multi_setting(args, "devices");
  } else if (std::getenv("HOUMO_VISIBLE_DEVICES") != nullptr) {
    std::unordered_map<std::string, std::string> device_args;
    device_args["devices"] = std::getenv("HOUMO_VISIBLE_DEVICES");
    devices = validate_multi_setting(device_args, "devices");
  } else {
    devices = {0};
  }
  if (devices.size() != 1) {
    throw std::invalid_argument("TTS perf supports exactly one device");
  }
  settings.device_id = devices.front();

  settings.loop =
      args.count("loop") ? ValidateTtsPositiveInteger(args, "loop") : 1;
  if (settings.loop > 1000000) {
    throw std::invalid_argument("loop must be in range [1, 1000000]");
  }
  settings.warm_up = !args.count("no_warm_up");
  settings.interval_ms = args.count("interval")
                             ? ValidateTtsPositiveInteger(args, "interval")
                             : 500;
  settings.model_name =
      args.count("model_name")
          ? args.at("model_name")
          : (args.count("ModelName") ? args.at("ModelName") : "qwen3-tts");
  if (settings.model_name.empty()) {
    throw std::invalid_argument("model_name must not be empty");
  }
  if (args.count("output_wav")) settings.output_wav = args.at("output_wav");
  if (args.count("dump_file")) settings.dump_file = args.at("dump_file");
  return settings;
}

static void PrintTtsSettings(const TtsPerfSettings& settings) {
  std::cout << COLOR_YELLOW << std::string(25, '=') << " TTS Perf Settings "
            << std::string(25, '=') << "\n"
            << "model: " << settings.model_name << "\n"
            << "mode: streaming fixed-frame\n"
            << "text_projection: " << settings.text_projection_path << "\n"
            << "talker_prefill: " << settings.talker_prefill_path << "\n"
            << "talker_decode: " << settings.talker_decode_path << "\n"
            << "code_predictor_prefill: "
            << settings.code_predictor_prefill_path << "\n"
            << "code_predictor_decode: " << settings.code_predictor_decode_path
            << "\n"
            << "stateful_decoder: " << settings.stateful_decoder_path << "\n"
            << "embedding: " << settings.embedding_path << "\n"
            << "code_embedding: " << settings.code_embedding_path << "\n"
            << "text_embedding: " << settings.text_embedding_path << "\n"
            << "requested_audio_length: " << std::fixed << std::setprecision(3)
            << settings.requested_audio_length_s << "s\n"
            << "token_per_second: " << settings.token_per_second << "\n"
            << "body_text_tokens: " << settings.body_text_tokens << "\n"
            << "text_projection_tokens: " << settings.text_projection_tokens
            << "\n"
            << "target_codec_frames: " << settings.target_codec_frames << "\n"
            << "nominal_audio_length: " << settings.nominal_audio_length_s
            << "s\n"
            << "expected_audio_samples: " << settings.expected_audio_samples
            << "\n"
            << "decoder_chunks: " << settings.decoder_chunks << "\n"
            << "devices: " << settings.device_id << "\n"
            << "loop: " << settings.loop << "\n"
            << "warm_up: " << (settings.warm_up ? "enable" : "disable") << "\n"
            << "interval: " << settings.interval_ms << " ms\n"
            << "output_wav: "
            << (settings.output_wav.empty() ? "disabled" : settings.output_wav)
            << "\n"
            << "dump_file: "
            << (settings.dump_file.empty() ? "disabled" : settings.dump_file)
            << "\n"
            << "seed: " << settings.seed << " (fixed)\n"
            << std::string(69, '=') << COLOR_RESET << std::endl;
}

class TtsProgressBar {
 public:
  explicit TtsProgressBar(std::string label) : label_(std::move(label)) {}

  void Update(size_t completed_frames, size_t total_frames) {
    if (total_frames == 0) return;
    const size_t filled =
        std::min(kBarWidth, completed_frames * kBarWidth / total_frames);
    if (filled == previous_filled_ && completed_frames != total_frames) return;
    previous_filled_ = filled;

    const int percent = static_cast<int>(completed_frames * 100 / total_frames);
    const double completed_audio = completed_frames * PerfTts::kSecondsPerFrame;
    const double total_audio = total_frames * PerfTts::kSecondsPerFrame;
    std::cout << '\r' << label_ << ": " << std::setw(3) << percent << "% |"
              << std::string(filled, '*')
              << std::string(kBarWidth - filled, ' ') << "| "
              << completed_frames << '/' << total_frames << " frames | "
              << std::fixed << std::setprecision(2) << completed_audio << '/'
              << total_audio << "s" << std::flush;
    if (completed_frames == total_frames) std::cout << '\n';
  }

 private:
  static constexpr size_t kBarWidth = 50;
  std::string label_;
  size_t previous_filled_ = std::numeric_limits<size_t>::max();
};

static void PrintTtsStageRow(const std::string& component, double time_ms,
                             size_t count, double e2e_ms) {
  const double percent = e2e_ms > 0.0 ? time_ms / e2e_ms * 100.0 : 0.0;
  std::cout << "| " << std::left << std::setw(28) << component << " | "
            << std::right << std::setw(10) << std::fixed << std::setprecision(2)
            << time_ms << " | " << std::setw(7) << percent << " | ";
  if (count == 0) {
    std::cout << std::setw(8) << "-"
              << " | " << std::setw(10) << "-";
  } else {
    std::cout << std::setw(8) << count << " | " << std::setw(10)
              << time_ms / count;
  }
  std::cout << " |\n";
}

static void PrintTtsMetrics(const TtsPerfResult& result,
                            const std::string& title) {
  std::cout << "\n================================ " << title
            << " ================================\n"
            << "Codec Frames:          " << result.generated_frames << "\n"
            << "Audio Samples:         " << result.audio_samples << "\n"
            << "Audio Duration:        " << std::fixed << std::setprecision(3)
            << result.audio_duration_s << " s\n"
            << "Decoder Chunks:        " << result.decoder_chunks << "\n"
            << "E2E:                   " << std::setprecision(2)
            << result.e2e_ms << " ms\n"
            << "RTF:                   " << std::setprecision(4) << result.rtf
            << (result.rtf < 1.0 ? " (< real-time)" : "") << "\n"
            << "TTFA:                  " << std::setprecision(2)
            << result.ttfa_ms << " ms\n"
            << "Codec Frames/s:        " << result.codec_frames_per_second
            << "\n"
            << "+------------------------------+------------+---------+--------"
               "--+------------+\n"
            << "| Stage                        |   Time(ms) |     Pct |    "
               "Count |    Avg(ms) |\n"
            << "+------------------------------+------------+---------+--------"
               "--+------------+\n";
  const auto& stage = result.stages;
  PrintTtsStageRow("text_embedding", stage.text_embedding_ms,
                   stage.text_embedding_count, result.e2e_ms);
  PrintTtsStageRow("text_projection", stage.text_projection_ms,
                   stage.text_projection_count, result.e2e_ms);
  PrintTtsStageRow("prompt_prepare", stage.prompt_prepare_ms,
                   stage.prompt_prepare_count, result.e2e_ms);
  PrintTtsStageRow("talker_prefill", stage.talker_prefill_ms,
                   stage.talker_prefill_count, result.e2e_ms);
  PrintTtsStageRow("talker_decode", stage.talker_decode_ms,
                   stage.talker_decode_count, result.e2e_ms);
  PrintTtsStageRow("talker_sampling", stage.talker_sampling_ms,
                   stage.talker_sampling_count, result.e2e_ms);
  PrintTtsStageRow("codec_frame_prepare", stage.codec_frame_prepare_ms, 0,
                   result.e2e_ms);
  PrintTtsStageRow("code_predictor_prepare", stage.code_predictor_prepare_ms, 0,
                   result.e2e_ms);
  PrintTtsStageRow("code_predictor_prefill", stage.code_predictor_prefill_ms,
                   stage.code_predictor_prefill_count, result.e2e_ms);
  PrintTtsStageRow("code_predictor_decode", stage.code_predictor_decode_ms,
                   stage.code_predictor_decode_count, result.e2e_ms);
  PrintTtsStageRow("code_predictor_sampling", stage.code_predictor_sampling_ms,
                   stage.code_predictor_sampling_count, result.e2e_ms);
  PrintTtsStageRow("stateful_decoder", stage.stateful_decoder_ms,
                   stage.stateful_decoder_count, result.e2e_ms);
  PrintTtsStageRow("other", stage.other_ms, 0, result.e2e_ms);
  std::cout << "+------------------------------+------------+---------+--------"
               "--+------------+\n";
}

static void AccumulateTtsResult(TtsPerfResult* total,
                                const TtsPerfResult& result) {
  if (total->generated_frames != result.generated_frames ||
      total->audio_samples != result.audio_samples ||
      total->decoder_chunks != result.decoder_chunks ||
      total->stages.text_embedding_count !=
          result.stages.text_embedding_count ||
      total->stages.text_projection_count !=
          result.stages.text_projection_count ||
      total->stages.prompt_prepare_count !=
          result.stages.prompt_prepare_count ||
      total->stages.talker_prefill_count !=
          result.stages.talker_prefill_count ||
      total->stages.talker_decode_count != result.stages.talker_decode_count ||
      total->stages.talker_sampling_count !=
          result.stages.talker_sampling_count ||
      total->stages.code_predictor_prefill_count !=
          result.stages.code_predictor_prefill_count ||
      total->stages.code_predictor_decode_count !=
          result.stages.code_predictor_decode_count ||
      total->stages.code_predictor_sampling_count !=
          result.stages.code_predictor_sampling_count ||
      total->stages.stateful_decoder_count !=
          result.stages.stateful_decoder_count) {
    throw std::runtime_error("TTS discrete results changed between loops");
  }

  total->e2e_ms += result.e2e_ms;
  total->ttfa_ms += result.ttfa_ms;
  total->rtf += result.rtf;
  total->codec_generation_ms += result.codec_generation_ms;
  total->codec_frames_per_second += result.codec_frames_per_second;
  total->stages.text_embedding_ms += result.stages.text_embedding_ms;
  total->stages.text_projection_ms += result.stages.text_projection_ms;
  total->stages.prompt_prepare_ms += result.stages.prompt_prepare_ms;
  total->stages.talker_prefill_ms += result.stages.talker_prefill_ms;
  total->stages.talker_decode_ms += result.stages.talker_decode_ms;
  total->stages.talker_sampling_ms += result.stages.talker_sampling_ms;
  total->stages.codec_frame_prepare_ms += result.stages.codec_frame_prepare_ms;
  total->stages.code_predictor_prepare_ms +=
      result.stages.code_predictor_prepare_ms;
  total->stages.code_predictor_prefill_ms +=
      result.stages.code_predictor_prefill_ms;
  total->stages.code_predictor_decode_ms +=
      result.stages.code_predictor_decode_ms;
  total->stages.code_predictor_sampling_ms +=
      result.stages.code_predictor_sampling_ms;
  total->stages.stateful_decoder_ms += result.stages.stateful_decoder_ms;
  total->stages.other_ms += result.stages.other_ms;
}

static TtsPerfResult AverageTtsResult(TtsPerfResult total, int count) {
  total.e2e_ms /= count;
  total.ttfa_ms /= count;
  total.codec_generation_ms /= count;
  total.stages.text_embedding_ms /= count;
  total.stages.text_projection_ms /= count;
  total.stages.prompt_prepare_ms /= count;
  total.stages.talker_prefill_ms /= count;
  total.stages.talker_decode_ms /= count;
  total.stages.talker_sampling_ms /= count;
  total.stages.codec_frame_prepare_ms /= count;
  total.stages.code_predictor_prepare_ms /= count;
  total.stages.code_predictor_prefill_ms /= count;
  total.stages.code_predictor_decode_ms /= count;
  total.stages.code_predictor_sampling_ms /= count;
  total.stages.stateful_decoder_ms /= count;
  total.stages.other_ms /= count;
  total.rtf = total.audio_duration_s > 0.0
                  ? total.e2e_ms / 1000.0 / total.audio_duration_s
                  : 0.0;
  total.codec_frames_per_second =
      total.codec_generation_ms > 0.0
          ? static_cast<double>(total.generated_frames) * 1000.0 /
                total.codec_generation_ms
          : 0.0;
  return total;
}

static void RunTtsWarmUp(PerfTts& model, DeviceMonitor& device_monitor) {
  std::cout << COLOR_BLUE << "\n"
            << std::string(30, '=') << " TTS Perf WarmUp "
            << std::string(30, '=') << "\n";
  std::cout << "Device temperature: " << device_monitor.getCurrentTemperature()
            << " C\n";
  TtsProgressBar progress("TTS WarmUp");
  model.Run(false, [&](size_t completed, size_t total) {
    progress.Update(completed, total);
  });
  std::cout << std::string(79, '=') << COLOR_RESET << "\n";
}

static void CheckTtsTemperature(float temperature) {
  std::cout << "Device temperature: " << temperature << " C\n";
  if (temperature > kTtsAlarmTemperatureThreshold &&
      temperature < kTtsShutdownTemperatureThreshold) {
    std::cout << COLOR_YELLOW
              << "Device temperature beyond 80.0 C, Temperature Warning!"
              << COLOR_RESET << std::endl;
  }
  if (temperature >= kTtsShutdownTemperatureThreshold) {
    throw std::runtime_error(
        "Device temperature beyond 100.0 C, Shutdown the demo!");
  }
}

static int RunTtsCore(const TtsPerfSettings& parsed_settings,
                      PerfDumper& perf_dumper) {
  auto device_monitor =
      std::make_unique<DeviceMonitor>(parsed_settings.interval_ms);
#if defined(__linux__)
  auto host_mem_monitor =
      std::make_unique<HostMonitor>(parsed_settings.interval_ms);
#endif
  bool device_monitor_started = false;
  std::unordered_map<int, DeviceStats> end_device_stats;
#if defined(__linux__)
  bool host_monitor_started = false;
#endif
  HostMemoryInfo host_mem_info{}, max_host_mem_info{};
  const auto stop_monitors = [&]() {
    if (device_monitor_started) {
      device_monitor->stop();
      end_device_stats = device_monitor->getFinalDeviceStats();
      device_monitor_started = false;
    }
#if defined(__linux__)
    if (host_monitor_started) {
      host_mem_monitor->stop();
      max_host_mem_info = host_mem_monitor->getFinalMemoryInfo();
      host_monitor_started = false;
    }
#endif
  };

  try {
    device_monitor->start();
    device_monitor_started = true;
#if defined(__linux__)
    host_mem_monitor->start();
    host_monitor_started = true;
#endif

    PerfTts model(parsed_settings);
    const TtsPerfSettings& settings = model.settings();
    const std::unordered_map<int, DeviceStats> post_init_dev_stats =
        device_monitor->getDeviceStats();
#if defined(__linux__)
    host_mem_info = host_mem_monitor->getCurrentMemoryInfo();
#endif
    PrintTtsSettings(settings);

    if (settings.warm_up) {
      RunTtsWarmUp(model, *device_monitor);
    }

    TtsPerfResult total_result;
    for (int i = 0; i < settings.loop; ++i) {
      std::cout << COLOR_BLUE << "\n"
                << std::string(24, '=') << " TTS Perf Loop: " << (i + 1) << "/"
                << settings.loop << " " << std::string(24, '=') << "\n";
      const float temperature = device_monitor->getCurrentTemperature();
      CheckTtsTemperature(temperature);

      const bool keep_waveform =
          i + 1 == settings.loop && !settings.output_wav.empty();
      TtsProgressBar progress("TTS Loop " + std::to_string(i + 1) + "/" +
                              std::to_string(settings.loop));
      TtsPerfResult result =
          model.Run(keep_waveform, [&](size_t completed, size_t total) {
            progress.Update(completed, total);
          });
      perf_dumper.writeTtsPerfBrief(settings, result, i + 1);

      if (i == 0) {
        total_result = result;
        total_result.waveform.clear();
      } else {
        AccumulateTtsResult(&total_result, result);
      }

      if (keep_waveform) {
        houmo::Qwen3TTSWavWriter writer(settings.output_wav,
                                        PerfTts::kSampleRate);
        writer.Write(result.waveform);
        writer.Close();
        std::cout << "WAV saved to: " << settings.output_wav << "\n";
      }
#if defined(__linux__)
      max_host_mem_info = host_mem_monitor->getMaxMemoryInfo();
#endif
      end_device_stats = device_monitor->getDeviceStats();
    }

    const TtsPerfResult average = AverageTtsResult(total_result, settings.loop);
    PrintTtsMetrics(average, "TTS Average");
    stop_monitors();
    perf_dumper.dumpTtsPerf(settings, average, host_mem_info, max_host_mem_info,
                            post_init_dev_stats, end_device_stats);
    perf_dumper.generateYamlFile();
    std::cout << COLOR_RESET;
    return 0;
  } catch (...) {
    stop_monitors();
    throw;
  }
}

static int RunTts(std::unordered_map<std::string, std::string> args,
                  PerfDumper& perf_dumper, bool run_perf_by_yaml) {
  try {
    const bool use_global_dump_file = args.erase("tts_global_dump_file") > 0;
    TtsPerfSettings settings = ParseTtsSettings(std::move(args));
    if (run_perf_by_yaml && !use_global_dump_file) {
      PerfDumper stream_dumper;
      if (!settings.dump_file.empty()) {
        stream_dumper.setYamlFile(settings.dump_file, false);
      }
      return RunTtsCore(settings, stream_dumper);
    }
    if (!settings.dump_file.empty()) {
      perf_dumper.setYamlFile(settings.dump_file, run_perf_by_yaml);
    }
    return RunTtsCore(settings, perf_dumper);
  } catch (const std::exception& e) {
    std::cerr << "TTS Perf Error: " << e.what() << std::endl;
    return 1;
  }
}

#endif  // RUN_TTS_H
