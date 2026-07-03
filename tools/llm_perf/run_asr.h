/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: run_asr.h
 * Description:
 *   ASR Performance Test Runner using PerfAsr with simulated data.
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

#ifndef RUN_ASR_H
#define RUN_ASR_H

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "asr/PerfAsr.h"
#include "utils/device_monitor/device_monitor.h"
#include "utils/perf_dumper/perf_dumper.h"
#include "utils/utils.h"
#if defined(__linux__)
#include "utils/host_monitor/host_monitor.h"
#endif

namespace fs = std::filesystem;

#define ALARM_TEMPERATURE_THRESHOLD 80
#define SHUTDOWN_TEMPERATURE_THRESHOLD 100

static void PrintAsrMetrics(const AsrTranscribeResult& result, int n_chunks) {
  double enc_chunk_s =
      (n_chunks > 0) ? result.encode_time_ms / 1000.0 / n_chunks : 0;
  double pref_chunk_s =
      (n_chunks > 0) ? result.prefill_time_ms / 1000.0 / n_chunks : 0;
  double dec_chunk_s =
      (n_chunks > 0) ? result.decode_time_ms / 1000.0 / n_chunks : 0;

  std::cout << "\n================================ ASR RTF Metrics "
            << "================================\n"
            << "Simulated Audio:      " << result.audio_duration_s << "s\n"
            << "Chunks:               " << n_chunks << "\n"
            << "Output Tokens:        " << result.output_tokens << "\n"
            << "Overall RTF:          " << std::fixed << std::setprecision(4)
            << result.overall_rtf
            << (result.overall_rtf < 1.0f ? " (< real-time)" : "") << "\n"
            << "Inference RTF:        " << result.inference_rtf << "\n"
            << "Encode Per Chunk:     " << std::fixed << std::setprecision(2)
            << enc_chunk_s << "s\n"
            << "Prefill Per Chunk:    " << pref_chunk_s << "s\n"
            << "Decode Per Chunk:     " << dec_chunk_s << "s\n"
            << "Decode TPS:           " << std::fixed << std::setprecision(2)
            << result.decode_tps << " tok/s\n"
            << "Overall TPS:          " << result.overall_tps << " tok/s\n"
            << "TTFT:                 " << std::fixed << std::setprecision(2)
            << result.ttft_ms << " ms\n"
            << "================================================================="
            << "=================\n";
}

static int RunAsrCore(const AsrPerfSettings& settings,
                      PerfDumper& perf_dumper) {
  auto device_monitor = std::make_unique<DeviceMonitor>(settings.interval_ms);
  device_monitor->start();
#if defined(__linux__)
  auto host_mem_monitor = std::make_unique<HostMonitor>(settings.interval_ms);
  host_mem_monitor->start();
#endif
  HostMemoryInfo host_mem_info{}, max_host_mem_info{};

  houmo::ModelConfig config;
  config.devices = settings.devices;
  config.prefill_path = settings.prefill_path;
  config.decode_path = settings.decode_path;
  config.extra_params["encoder_path"] = settings.encode_path;

  auto model = std::make_unique<PerfAsrModel>(config);
  auto ctx = model->create_context();
  auto* perf_ctx = static_cast<PerfAsrContext*>(ctx.get());
  std::unordered_map<int, DeviceStats> post_init_dev_stats =
      device_monitor->getDeviceStats();
#if defined(__linux__)
  host_mem_info = host_mem_monitor->getCurrentMemoryInfo();
#endif

  int encoder_window = model->encoder_window();
  int n_frames = static_cast<int>(settings.audio_len_seconds * 16000 / 160);
  if (n_frames < 1) n_frames = 1;
  int n_chunks = (n_frames + encoder_window - 1) / encoder_window;

  if (settings.warm_up) {
    std::cout << "\n"
              << std::string(30, '=') << "ASR Perf WarmUp: audio_len="
              << settings.audio_len_seconds
              << "s token_per_second=" << settings.token_per_second
              << std::string(30, '=') << "\n";
    float temp = device_monitor->getCurrentTemperature();
    std::cout << "Device temperature: " << temp << " C" << std::endl;
    perf_ctx->PerfRun(settings.audio_len_seconds, settings.token_per_second);
    perf_ctx->profiler().print_summary();
    std::cout << std::string(82, '=') << "\n";
  }

  for (int i = 0; i < settings.loop_count; ++i) {
    std::cout << COLOR_BLUE << "\n"
              << std::string(24, '=') << "ASR Perf Loop: " << (i + 1) << "/"
              << settings.loop_count
              << " | audio_len=" << settings.audio_len_seconds
              << "s | token_per_second=" << settings.token_per_second
              << std::string(24, '=') << "\n";

    float temp = device_monitor->getCurrentTemperature();
    std::cout << "Device temperature: " << temp << " C" << std::endl;
    if (temp > ALARM_TEMPERATURE_THRESHOLD &&
        temp < SHUTDOWN_TEMPERATURE_THRESHOLD) {
      std::cout << COLOR_YELLOW
                << "Device temperature beyond 80.0 C, Temperature Warning!"
                << COLOR_RESET << std::endl;
    }
    if (temp >= SHUTDOWN_TEMPERATURE_THRESHOLD) {
      throw std::runtime_error(
          "Device temperature beyond 100.0 C, Shutdown the demo!");
    }

    auto result =
        perf_ctx->PerfRun(settings.audio_len_seconds, settings.token_per_second);
    PrintAsrMetrics(result, n_chunks);
    perf_ctx->profiler().print_summary();
#if defined(__linux__)
    max_host_mem_info = host_mem_monitor->getMaxMemoryInfo();
#endif
    std::unordered_map<int, DeviceStats> current_dev_stats =
        device_monitor->getDeviceStats();
    perf_dumper.dumpAsrPerf(settings, result, n_chunks, host_mem_info,
                            max_host_mem_info, post_init_dev_stats,
                            current_dev_stats);
    std::cout << std::string(82, '=') << "\n";
  }

  ctx.reset();
  model.reset();
  device_monitor->stop();
  std::unordered_map<int, DeviceStats> end_dev_stats =
      device_monitor->getFinalDeviceStats();
#if defined(__linux__)
  host_mem_monitor->stop();
  max_host_mem_info = host_mem_monitor->getFinalMemoryInfo();
#endif
  perf_dumper.generateYamlFile();

  return 0;
}

static int RunAsr(std::unordered_map<std::string, std::string> args,
                  PerfDumper& perf_dumper,
                  bool run_perf_by_yaml) {
  try {
    AsrPerfSettings settings;

    fs::path encode_path = validate_path(args, "encode");
    fs::path prefill_path = validate_path(args, "prefill");
    fs::path decode_path = validate_path(args, "decode");
    settings.encode_path = encode_path.string();
    settings.prefill_path = prefill_path.string();
    settings.decode_path = decode_path.string();

    if (args.find("model_name") != args.end()) {
      settings.model_name = args["model_name"];
    } else if (args.find("ModelName") != args.end()) {
      settings.model_name = args["ModelName"];
    }

    if (args.count("devices")) {
      std::unordered_map<std::string, std::string> tmp;
      tmp["devices"] = args["devices"];
      settings.devices = validate_multi_setting(tmp, "devices");
    } else if (getenv("HOUMO_VISIBLE_DEVICES") != nullptr) {
      std::unordered_map<std::string, std::string> tmp;
      tmp["devices"] = getenv("HOUMO_VISIBLE_DEVICES");
      settings.devices = validate_multi_setting(tmp, "devices");
    } else {
      settings.devices = {0};
    }

    settings.audio_len_seconds =
        args.count("audio_len") ? std::stof(args["audio_len"]) : 30.0f;

    settings.token_per_second =
        args.count("token_per_second")
            ? validate_setting(args, "token_per_second")
            : 20;

    settings.loop_count =
        args.count("loop") ? validate_setting(args, "loop") : 1;
    settings.loop_count =
        std::min(std::max(settings.loop_count, 1), 1000000);

    settings.warm_up = !args.count("no_warm_up");

    settings.interval_ms =
        args.count("interval") ? validate_setting(args, "interval") : 500;

    if (args.count("dump_file") == 1) {
      perf_dumper.setYamlFile(args["dump_file"], run_perf_by_yaml);
    }

    std::cout << COLOR_YELLOW << std::string(25, '=')
              << " ASR Perf Settings " << std::string(25, '=') << "\n"
              << "model: " << settings.model_name << "\n"
              << "encode: " << settings.encode_path << "\n"
              << "prefill: " << settings.prefill_path << "\n"
              << "decode: " << settings.decode_path << "\n"
              << "audio_len: " << settings.audio_len_seconds << "s\n"
              << "token_per_second: " << settings.token_per_second << "\n"
              << "devices: " << format_int_list(settings.devices) << "\n"
              << "loop: " << settings.loop_count << "\n"
              << "warm_up: " << (settings.warm_up ? "enable" : "disable")
              << "\n"
              << std::string(65, '=') << COLOR_RESET << std::endl;

    return RunAsrCore(settings, perf_dumper);
  } catch (const std::exception& e) {
    std::cerr << "ASR Perf Error: " << e.what() << std::endl;
    return 1;
  }
}

static int RunAsrConfig(int argc, char* argv[], PerfDumper& perf_dumper) {
  const std::string yamlfile = argv[2];
  fs::path path = fs::u8path(yamlfile);
  if (!fs::exists(path)) {
    throw std::invalid_argument(
        "config path does not exist: " + path.u8string());
  }

  std::ifstream file(yamlfile);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string yaml_content = buffer.str();

  YAML::Node config = YAML::Load(yaml_content);
  if (!config["Streams"]) {
    throw std::invalid_argument(
        "config file does not contain perf Streams!");
  }

  size_t n_tasks = config["Streams"].size();
  size_t curTaskId = 0;

  if (config["dump_file"]) {
    std::string dump_file = config["dump_file"].as<std::string>();
    perf_dumper.setYamlFile(dump_file, true);
    std::cout << COLOR_GREEN << "Dump perf to file: " << dump_file << "\n";
  }

  for (const auto& stream : config["Streams"]) {
    std::string model_name = stream["ModelName"]
                                 ? stream["ModelName"].as<std::string>()
                                 : "unknown";

    std::cout << COLOR_GREEN << std::string(45, '#') << "Start of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ", ModelName:" << model_name << "."
              << std::string(45, '#') << "\n";

    AsrPerfSettings settings;
    settings.model_name = model_name;
    settings.encode_path = stream["encode"].as<std::string>();
    settings.prefill_path = stream["prefill"].as<std::string>();
    settings.decode_path = stream["decode"].as<std::string>();
    settings.audio_len_seconds =
        stream["audio_len"] ? stream["audio_len"].as<float>() : 30.0f;
    settings.token_per_second =
        stream["token_per_second"]
            ? stream["token_per_second"].as<int>()
            : 20;

    if (stream["devices"]) {
      std::string dev_str = stream["devices"].as<std::string>();
      std::unordered_map<std::string, std::string> tmp;
      tmp["devices"] = dev_str;
      settings.devices = validate_multi_setting(tmp, "devices");
    } else {
      settings.devices = {0};
    }

    settings.loop_count =
        stream["loop"] ? stream["loop"].as<int>() : 1;
    settings.warm_up =
        stream["no_warm_up"] ? false : true;
    settings.interval_ms =
        stream["interval"] ? stream["interval"].as<int>() : 500;

    if (!config["dump_file"] && stream["dump_file"]) {
      std::string dump_file = stream["dump_file"].as<std::string>();
      perf_dumper.setYamlFile(dump_file, false);
      std::cout << COLOR_GREEN << "Dump perf to file: " << dump_file << "\n";
    }

    RunAsrCore(settings, perf_dumper);

    std::cout << COLOR_GREEN << std::string(45, '#') << " End of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ",  ModelName:" << model_name << "."
              << std::string(45, '#') << "\n\n\n";
    curTaskId++;
  }

  std::cout << COLOR_RESET;
  return 0;
}

#endif  // RUN_ASR_H
