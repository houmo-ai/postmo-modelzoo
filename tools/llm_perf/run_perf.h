/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: run_perf.h
 * Description:
 *   Run LLM Model Functions - Functions for running performance
 * tests on large language models with various configurations.
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
#include <yaml-cpp/yaml.h>

#include <codecvt>
#include <filesystem>
#include <iostream>
#include <locale>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "HmllmInfer.h"
#include "HmllmInferMultiBatch.h"
#include "HmvllmInfer.h"
#include "device_monitor/device_monitor.h"
#include "perf_dumper/perf_dumper.h"
#include "tcim/tcim_runtime.h"
#include "utils.h"
#if defined(__linux__)
#include "host_monitor/host_monitor.h"
#endif

#ifdef _MSC_VER
#include <Windows.h>
#endif

#define ALARM_TEMPERATURE_THRESHOLD 80
#define SHUTDOWN_TEMPERATURE_THRESHOLD 100

static PerfDumper perf_dumper = PerfDumper();
static bool run_perf_by_yaml = false;
PerfSettings ParsePerfRunSetting(
    std::unordered_map<std::string, std::string> args) {
  PerfSettings settings;
  if (args.find("ModelName") != args.end()) {
    settings.model_name = args["ModelName"];
  }

  fs::path prefill_path = validate_path(args, "prefill");
  fs::path decode_path = validate_path(args, "decode");
  fs::path visual_path =
      args.count("visual") ? validate_path(args, "visual") : fs::path();
  fs::path embedding_path = validate_path(args, "embedding");
  int input_token_len = validate_setting(args, "input");
  int stop_token_len = validate_setting(args, "output");
  int ndevices =
      args.count("ndevices") ? validate_setting(args, "ndevices") : 1;
  if (ndevices > tcim::GetDeviceNum() || ndevices < 1) {
    throw std::invalid_argument("ndevices must <= device number and >= 1");
  }

  bool skip_perf = args.count("skip_perf") ? true : false;
  int loop_round = args.count("loop") ? validate_setting(args, "loop") : 1;
  loop_round = std::min(std::max(loop_round, 1), 1000000);
  int batch = args.count("batch") ? validate_setting(args, "batch") : 1;
  bool warm_up_enable = args.count("no_warm_up") ? false : true;
  bool lazy_mode_enable = args.count("LazyMode") ? true : false;

  std::cout << COLOR_YELLOW << std::string(25, '=') << " Perf Settings "
            << std::string(25, '=') << std::endl;
  std::cout << "prefill path : " << prefill_path.string() << std::endl;
  std::cout << "decode path : " << decode_path.string() << std::endl;
  if (!visual_path.empty()) {
    std::cout << "visual path : " << visual_path.string() << std::endl;
  }
  std::cout << "embedding path : " << embedding_path.string() << std::endl;
  std::cout << "input token len : " << input_token_len << std::endl;
  std::cout << "stop token len : " << stop_token_len << std::endl;
  std::cout << "ndevices : " << ndevices << std::endl;
  std::cout << "loop : " << loop_round << std::endl;
  std::cout << "batch : " << batch << std::endl;
  uint32_t warm_up_input = 0;
  uint32_t warm_up_output = 0;
  if (warm_up_enable) {
    std::cout << "warm_up : enable" << std::endl;
    warm_up_input = args.count("warm_up_input")
                        ? validate_setting(args, "warm_up_input")
                        : input_token_len;
    warm_up_output = args.count("warm_up_output")
                         ? validate_setting(args, "warm_up_output")
                         : stop_token_len;
  } else {
    std::cout << "warm_up : disable" << std::endl;
  }

  if (lazy_mode_enable) {
    std::cout << "LazyMode : enable (this may lead to loading model taking "
                 "more time)"
              << std::endl;
  } else {
    std::cout << "LazyMode : disable" << std::endl;
  }

  if (skip_perf) {
    std::cout << "skip_perf : enable" << std::endl;
  } else {
    std::cout << "skip_perf : disable" << std::endl;
  }

  std::cout << std::string(65, '=') << COLOR_RESET << std::endl;

  settings.prefill_path = prefill_path.string();
  settings.decode_path = decode_path.string();
  settings.embedding_path = embedding_path.string();
  settings.visual_path = visual_path.string();
  settings.input_tokens_len = input_token_len;
  settings.stop_tokens_len = stop_token_len;
  settings.ndevices = ndevices;
  settings.batch_size = batch;
  settings.LazyMode = lazy_mode_enable;
  settings.warm_up = warm_up_enable;
  settings.warm_up_input = warm_up_input;
  settings.warm_up_output = warm_up_output;
  settings.loop_count = loop_round;
  settings.skip_perf = skip_perf;

  return settings;
}

int RunPerf(std::unordered_map<std::string, std::string> args) {
  int interval =
      args.count("interval") ? validate_setting(args, "interval") : 500;
  interval = std::min(std::max(interval, 100), 60000);
  auto device_monitor = std::make_unique<DeviceMonitor>(interval);
  device_monitor->start();
#if defined(__linux__)
  auto host_mem_monitor = std::make_unique<HostMonitor>(interval);
  host_mem_monitor->start();
#endif
  HostMemoryInfo host_mem_info, max_host_mem_info;
  if (args.count("dump_file") == 1) {
    perf_dumper.setYamlFile(args["dump_file"], run_perf_by_yaml);
  }
  try {
    PerfSettings settings = ParsePerfRunSetting(args);
    settings.interval_ms = interval;
    const char* houmo_target_env = getenv("HOUMO_TARGET");
    std::string houmo_target =
        houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
    if (houmo_target != "xh2") {
      throw std::invalid_argument("Unsupported backend " + houmo_target);
    }

    std::unordered_map<int, DeviceStats> start_dev_stats =
        device_monitor->getDeviceStats();
    std::unique_ptr<HmllmInferBase> Qwen3Infer;
    if (settings.visual_path.empty()) {
      if (settings.batch_size == 1) {
        Qwen3Infer = std::make_unique<HmllmInfer>(
            settings.prefill_path, settings.decode_path,
            settings.embedding_path, settings.ndevices, settings.batch_size,
            settings.LazyMode);
      } else {
        if (houmo_target != "xh2") {
          throw std::runtime_error(
              "Only xh2 support multibacth, device not match!");
        }
        Qwen3Infer = std::make_unique<HmllmInferMultiBatch>(
            settings.prefill_path, settings.decode_path,
            settings.embedding_path, settings.ndevices, settings.batch_size,
            settings.LazyMode);
      }
    } else {
      Qwen3Infer = std::make_unique<HmvllmInfer>(
          settings.prefill_path, settings.decode_path, settings.embedding_path,
          settings.visual_path, settings.ndevices, settings.batch_size,
          settings.LazyMode);
    }

#if defined(__linux__)
    host_mem_info = host_mem_monitor->getCurrentMemoryInfo();
#endif
    std::unordered_map<int, DeviceStats> end_dev_stats =
        device_monitor->getDeviceStats();

    if (!settings.skip_perf) {
      if (settings.warm_up) {
        std::cout << "\n"
                  << std::string(30, '=') << "(v)LLM Perf WarmUp: input "
                  << settings.warm_up_input << ", output "
                  << settings.warm_up_output << std::string(30, '=') << "\n ";
        float current_temperature = device_monitor->getMaxTemperature();
        std::cout << "Device temperature: " << current_temperature << " °C"
                  << std::endl;
        if (current_temperature > ALARM_TEMPERATURE_THRESHOLD &&
            current_temperature < SHUTDOWN_TEMPERATURE_THRESHOLD) {
          std::cout
              << COLOR_YELLOW
              << "Device temperature is beyond 80.0 °C, Temperature Warning!"
              << COLOR_RESET << std::endl;
        }
        if (current_temperature >= SHUTDOWN_TEMPERATURE_THRESHOLD) {
          throw std::runtime_error(
              "Device temperature is beyond 100.0 °C, "
              "Shutdown the demo!");
        }
        Qwen3Infer->get_perf_tracker()->reset();
        Qwen3Infer->perf_llm(settings.warm_up_input, settings.warm_up_output);
        Qwen3Infer->get_perf_tracker()->pref_delete_warmup();
        std::cout << "\n" << std::string(82, '=') << "\n";
#if defined(__linux__)
        max_host_mem_info = host_mem_monitor->getMaxMemoryInfo();
#endif
        InferenceMetricsWithLoadTime perf_metrics =
            Qwen3Infer->get_perf_tracker()->get_perf_current_summary();
        perf_dumper.writePerfBrief(settings, perf_metrics, host_mem_info,
                                   max_host_mem_info, start_dev_stats,
                                   end_dev_stats, "llm-perf warmup");
      }

      for (int i = 0; i < settings.loop_count; ++i) {
        std::cout << COLOR_BLUE << "\n"
                  << std::string(30, '=')
                  << "(v)LLM Perf Loop Progress: " << (i + 1) << "/"
                  << settings.loop_count << std::string(30, '=') << "\n ";
        float current_temperature = device_monitor->getMaxTemperature();
        std::cout << "Device temperature: " << current_temperature << " °C"
                  << std::endl;
        if (current_temperature > ALARM_TEMPERATURE_THRESHOLD &&
            current_temperature < SHUTDOWN_TEMPERATURE_THRESHOLD) {
          std::cout
              << COLOR_YELLOW
              << "Device temperature is beyond 80.0 °C, Temperature Warning!"
              << COLOR_RESET << std::endl;
        }
        if (current_temperature >= SHUTDOWN_TEMPERATURE_THRESHOLD) {
          throw std::runtime_error(
              "Device temperature is beyond 100.0 °C, "
              "Shutdown the demo!");
        }
        Qwen3Infer->get_perf_tracker()->reset();
        Qwen3Infer->perf_llm(settings.input_tokens_len,
                             settings.stop_tokens_len);
        std::cout << "\n" << std::string(82, '=') << "\n";
#if defined(__linux__)
        max_host_mem_info = host_mem_monitor->getMaxMemoryInfo();
#endif
        InferenceMetricsWithLoadTime perf_metrics =
            Qwen3Infer->get_perf_tracker()->get_perf_current_summary();
        perf_dumper.writePerfBrief(
            settings, perf_metrics, host_mem_info, max_host_mem_info,
            start_dev_stats, end_dev_stats,
            "llm-perf Loop Progress: " + std::to_string(i + 1) + "/" +
                std::to_string(settings.loop_count));
      }

      Qwen3Infer->get_perf_tracker()->showSummary(true);
      std::cout << COLOR_RESET;
    }
    InferenceMetricsWithLoadTime metrics =
        Qwen3Infer->get_perf_tracker()->get_perf_avg_summary();

    Qwen3Infer.reset();
    device_monitor->stop();

#if defined(__linux__)
    host_mem_monitor->stop();
    max_host_mem_info = host_mem_monitor->getFinalMemoryInfo();
#endif
    perf_dumper.dumpPerf(settings, metrics, host_mem_info, max_host_mem_info,
                         start_dev_stats, end_dev_stats);
    perf_dumper.showPerfBrief(settings, metrics, host_mem_info,
                              max_host_mem_info, start_dev_stats,
                              end_dev_stats);
    perf_dumper.writePerfBrief(settings, metrics, host_mem_info,
                               max_host_mem_info, start_dev_stats,
                               end_dev_stats, "llm-perf Average");
    perf_dumper.generateYamlFile();

    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    device_monitor->stop();
#if defined(__linux__)
    host_mem_monitor->stop();
#endif
    return 1;
  }

  device_monitor->cleanup();
  return 0;
}

int RunPerfConfig(int argc, char* argv[]) {
  const std::string yamlfile = argv[2];
  fs::path path = fs::u8path(yamlfile);
  if (!fs::exists(path)) {
    throw std::invalid_argument("config path does not exist: " +
                                path.u8string());
  }

  std::ifstream file(yamlfile);
  std::stringstream buffer;
  buffer << file.rdbuf();
  std::string yaml_content = buffer.str();

  YAML::Node config = YAML::Load(yaml_content);
  if (!config["Streams"]) {
    throw std::invalid_argument("config file does not contain perf Streams!");
  }

  size_t n_tasks = config["Streams"].size();
  size_t curTaskId = 0;

  if (config["dump_file"]) {
    run_perf_by_yaml = true;
    std::string dump_file = config["dump_file"].as<std::string>();
    perf_dumper.setYamlFile(dump_file, run_perf_by_yaml);
    std::cout << COLOR_GREEN << "Dump perf to file: " << dump_file << "\n";
  }

  for (const auto& stream : config["Streams"]) {
    std::cout << COLOR_GREEN << std::string(45, '#') << "Start of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ", ModelName:" << stream["ModelName"].as<std::string>() << "."
              << std::string(45, '#') << "\n";

    // Convert YAML node to unordered_map<string, string>
    std::unordered_map<std::string, std::string> args;
    for (const auto& kv : stream) {
      std::string key = kv.first.as<std::string>();
      std::string value = kv.second.as<std::string>();
      args[key] = value;
    }

    RunPerf(args);

    std::cout << COLOR_GREEN << std::string(45, '#') << " End of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ",  ModelName:" << stream["ModelName"].as<std::string>() << "."
              << std::string(45, '#') << "\n\n\n";
    curTaskId++;
  }

  std::cout << COLOR_RESET;
  return 0;
}