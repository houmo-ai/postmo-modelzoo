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
#include <codecvt>
#include <filesystem>
#include <iostream>
#include <locale>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "HmllmInfer.h"
#include "HmllmInferMultiBatch.h"
#include "HmvllmInfer.h"
#include "devices_monitor.h"
#include "perf_dumper/perf_dumper.h"
#include "tcim/tcim_runtime.h"
#include "utils.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

#define ALARM_TEMPERATURE_THRESHOLD 80
#define SHUTDOWN_TEMPERATURE_THRESHOLD 100

#ifdef PERF_DUMP_ENABLE
static PerfDumper perf_dumper = PerfDumper();
static bool run_perf_by_json = false;
#endif
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
  int stop_token_len = validate_setting(args, "stop");
  int ndevices =
      args.count("ndevices") ? validate_setting(args, "ndevices") : 1;
  int loop_round = args.count("loop") ? validate_setting(args, "loop") : 1;

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
  if (warm_up_enable) {
    std::cout << "warm_up : enable" << std::endl;
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
  settings.loop_count = loop_round;

  return settings;
}

int RunPerf(std::unordered_map<std::string, std::string> args) {
  int interval =
      args.count("interval") ? validate_setting(args, "interval") : 1;
  std::thread device_monitor_thread(device_monitor, 0, interval);
#ifdef PERF_DUMP_ENABLE
  if (args.count("dump_file") == 1) {
    perf_dumper.setJsonFile(args["dump_file"], run_perf_by_json);
  }
#endif
  try {
    PerfSettings settings = ParsePerfRunSetting(args);
    const char* houmo_target_env = getenv("HOUMO_TARGET");
    std::string houmo_target =
        houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
    if (houmo_target != "xh2") {
      throw std::invalid_argument("Unsupported backend " + houmo_target);
    }

#ifdef XH2A_HM_SYS
    std::map<int, hm_mem_info> dev_mem_info_start;
    auto mem_ret_start = GetDevMemInfo(dev_mem_info_start);
#endif

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

#ifdef XH2A_HM_SYS
    std::map<int, hm_mem_info> dev_mem_info_end;
    auto mem_ret_end = GetDevMemInfo(dev_mem_info_end);
    if (mem_ret_start == 0 && mem_ret_end == 0) {
      std::cout << "****** HM Device Memory Usage ******" << std::endl;
      for (const auto& pair : dev_mem_info_start) {
        int device_id = pair.first;
        const hm_mem_info& mem_info_start = pair.second;
        if (dev_mem_info_end.count(device_id) == 0) {
          std::cerr << "Failed to get device " << device_id << " memory info."
                    << std::endl;
          break;
        }
        const hm_mem_info& mem_info_end = dev_mem_info_end[device_id];
        int32_t mem_used = mem_info_end.mem_used - mem_info_start.mem_used;
        mem_used = mem_used < 0 ? 0 : mem_used;
        std::cout << "Device id: " << device_id << ", memory used: " << mem_used
                  << " MB" << std::endl;
      }
      std::cout << "************************************" << std::endl;
    } else {
      std::cerr << "Failed to get device memory info, start ret is "
                << mem_ret_start << ", end ret is " << mem_ret_end << std::endl;
    }
#endif
    if (settings.warm_up) {
      std::cout << "\n"
                << std::string(30, '=') << "(v)LLM Perf WarmUp: input "
                << settings.input_tokens_len << ", output "
                << settings.stop_tokens_len << std::string(30, '=') << "\n ";
      float current_temperature = get_temperature();
      std::cout << "Device temperature: " << current_temperature << " °C"
                << std::endl;
      if (current_temperature > ALARM_TEMPERATURE_THRESHOLD &&
          current_temperature < SHUTDOWN_TEMPERATURE_THRESHOLD) {
        std::cout
            << COLOR_YELLOW
            << "Device temperature is beyond 80.0 °C, Temperature Warning!"
            << std::endl;
      }
      if (current_temperature >= SHUTDOWN_TEMPERATURE_THRESHOLD) {
        throw std::runtime_error(
            "Device temperature is beyond 100.0 °C, "
            "Shutdown the demo!");
      }
      Qwen3Infer->get_perf_tracker()->reset();
      Qwen3Infer->perf_llm(settings.input_tokens_len, settings.stop_tokens_len);
      Qwen3Infer->get_perf_tracker()->pref_delete_warmup();
      std::cout << std::string(82, '=') << "\n";
    }
    get_mem_info(0);
    for (int i = 0; i < settings.loop_count; ++i) {
      std::cout << COLOR_BLUE << "\n"
                << std::string(30, '=')
                << "(v)LLM Perf Loop Progress: " << (i + 1) << "/"
                << settings.loop_count << std::string(30, '=') << "\n ";
      float current_temperature = get_temperature();
      std::cout << "Device temperature: " << current_temperature << " °C"
                << std::endl;
      if (current_temperature > ALARM_TEMPERATURE_THRESHOLD &&
          current_temperature < SHUTDOWN_TEMPERATURE_THRESHOLD) {
        std::cout
            << COLOR_YELLOW
            << "Device temperature is beyond 80.0 °C, Temperature Warning!"
            << std::endl;
      }
      if (current_temperature >= SHUTDOWN_TEMPERATURE_THRESHOLD) {
        throw std::runtime_error(
            "Device temperature is beyond 100.0 °C, "
            "Shutdown the demo!");
      }
      Qwen3Infer->get_perf_tracker()->reset();
      Qwen3Infer->perf_llm(settings.input_tokens_len, settings.stop_tokens_len);
      std::cout << std::string(82, '=') << "\n";
    }

    std::cout << COLOR_GREEN << std::string(30, '=')
              << " (v)LLM Perf Avarage Information " << std::string(30, '=')
              << "\n";
    Qwen3Infer->get_perf_tracker()->showSummary(true);
    std::cout << COLOR_GREEN << std::string(90, '=') << "\n";
    std::cout << COLOR_RESET;
    InferenceMetricsWithLoadTime metrics =
        Qwen3Infer->get_perf_tracker()->get_perf_avg_summary();
#ifdef PERF_DUMP_ENABLE
    perf_dumper.dumpPerf(settings, metrics);
#endif
    Qwen3Infer.reset();
    stop_monitor(0);
    device_monitor_thread.join();
#ifdef PERF_DUMP_ENABLE
    if (!run_perf_by_json) {
      perf_dumper.generateJsonFile();
    }
#endif
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    stop_monitor(0);
    device_monitor_thread.join();
    return 1;
  }
  return 0;
}

int RunPerfJson(int argc, char* argv[]) {
  const std::string jsonfile = argv[2];
  fs::path path = fs::u8path(jsonfile);
  if (!fs::exists(path)) {
    throw std::invalid_argument("config path does not exist: " +
                                path.u8string());
  }

  std::ifstream f(jsonfile);
  json perf_configs;
  f >> perf_configs;
  if (!perf_configs.contains("Streams")) {
    throw std::invalid_argument("config file does not contain perf Streams!");
  }
  int n_tasks = perf_configs["Streams"].size();
  int curTaskId = 0;
#ifdef PERF_DUMP_ENABLE
  if (perf_configs.contains("dump_file")) {
    run_perf_by_json = true;
    perf_dumper.setJsonFile(perf_configs["dump_file"], run_perf_by_json);
    std::cout << COLOR_GREEN
              << "Dump perf to file: " << perf_configs["dump_file"] << "\n";
  }
#endif
  for (json& stream : perf_configs["Streams"]) {
    std::cout << COLOR_GREEN << std::string(45, '#') << "Start of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ", ModelName:" << stream["ModelName"] << "."
              << std::string(45, '#') << "\n";

    std::unordered_map<std::string, std::string> args = parse_json(stream);
    RunPerf(args);
    std::cout << COLOR_GREEN << std::string(45, '#') << " End of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ",  ModelName:" << stream["ModelName"] << "."
              << std::string(45, '#') << "\n\n\n";
    curTaskId++;
  }
#ifdef PERF_DUMP_ENABLE
  perf_dumper.generateJsonFile();
#endif
  std::cout << COLOR_RESET;
  return 0;
}
