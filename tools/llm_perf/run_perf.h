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

static PerfSettings BuildPerfCaseSettings(
    const PerfSettings& settings, const PerfSettings::PerfCase& perf_case,
    size_t perf_case_index) {
  PerfSettings current_settings = settings;
  current_settings.input_tokens_len = perf_case.input_tokens_len;
  current_settings.stop_tokens_len = perf_case.stop_tokens_len;
  current_settings.perf_case_index = static_cast<int>(perf_case_index + 1);
  current_settings.perf_case_total =
      static_cast<int>(settings.perf_cases.size());
  return current_settings;
}

static void device_ctc_check(std::vector<int> devices,
                             const fs::path& prefill_path,
                             const fs::path& decode_path,
                             const fs::path& visual_path) {
  assert(prefill_path.extension() == decode_path.extension());
  const bool has_visual = !visual_path.string().empty();
  const bool llm_uses_hmms = (prefill_path.extension() == ".hmms") ||
                             (decode_path.extension() == ".hmms");
  const bool visual_uses_hmms =
      has_visual && visual_path.extension() == ".hmms";
  if (llm_uses_hmms) {
    if ((prefill_path.extension() != ".hmms") ||
        (decode_path.extension() != ".hmms") || devices.size() < 2) {
      throw std::invalid_argument(
          "For multi-device .hmms execution, prefill and decode model files "
          "must be .hmms format and devices must be at least 2. The visual "
          "model may be omitted, be .hmms, or be a single-device model "
          "loaded on the first device.");
    } else {
      int group_id, chip_id;
      std::vector<DeviceCtcInfo> device_ctc_info_list;
      for (int dev_id : devices) {
        if (hm_sys_get_ctc_phy_id(dev_id, &group_id, &chip_id) != 0) {
          throw std::runtime_error("Failed to get physical ID for device " +
                                   std::to_string(dev_id));
        }
        if (group_id < 0 || chip_id < 0) {
          throw std::runtime_error("Invalid physical ID for device " +
                                   std::to_string(dev_id));
        }
        std::cout << "Device " << dev_id << " -> Group " << group_id
                  << ", Chip " << chip_id << std::endl;
        device_ctc_info_list.push_back({dev_id, group_id, chip_id});
      }

      for (auto ctcInfo : device_ctc_info_list) {
        if (ctcInfo.group_id != device_ctc_info_list[0].group_id) {
          throw std::runtime_error(
              "All devices must be on the same group for multi-device "
              "testing with .hmms model files.");
        }
      }
    }
  } else if (visual_uses_hmms) {
    throw std::invalid_argument(
        "A .hmms visual model requires .hmms prefill/decode models.");
  }
  std::cout << "Device CTC check passed for devices: "
            << format_int_list(devices) << std::endl;
}

PerfSettings ParsePerfRunSetting(
    std::unordered_map<std::string, std::string> args) {
  PerfSettings settings;
  if (args.find("model_name") != args.end()) {
    settings.model_name = args["model_name"];
  } else if (args.find("ModelName") != args.end()) {
    settings.model_name = args["ModelName"];
  }

  fs::path prefill_path = validate_path(args, "prefill");
  fs::path decode_path = validate_path(args, "decode");
  fs::path visual_path =
      args.count("visual") ? validate_path(args, "visual") : fs::path();
  fs::path embedding_path = validate_path(args, "embedding");
  std::vector<int> input_token_lens = validate_multi_setting(args, "input");
  std::vector<int> stop_token_lens = validate_multi_setting(args, "output");
  if (input_token_lens.size() != stop_token_lens.size()) {
    throw std::invalid_argument(
        "input and output must have the same number of comma-separated "
        "values");
  }
  if (args.count("ndevices")) {
    throw std::invalid_argument(
        "The argument 'ndevices' is deprecated. Please use 'devices' with "
        "comma-separated device IDs instead (e.g., --devices 0,1,2).");
  }
  std::vector<int> devices;
  if (args.count("devices")) {
    devices = validate_multi_setting(args, "devices");
  } else if (getenv("HOUMO_VISIBLE_DEVICES") != nullptr) {
    std::unordered_map<std::string, std::string> temp_args;
    temp_args["devices"] = getenv("HOUMO_VISIBLE_DEVICES");
    devices = validate_multi_setting(temp_args, "devices");
    if (devices.size() > 1 && prefill_path.extension() != ".hmms") {
      std::cout << "[Warning] : "
                << "Multiple devices specified in HOUMO_VISIBLE_DEVICES, "
                   "but model file is not .hmms. Multi-device testing is "
                   "only supported for .hmms model files. Ignoring "
                   "additional devices and using HOUMO_VISIBLE_DEVICES[0] only."
                << std::endl;
      devices = {devices[0]};
    }
  } else {
    devices = {0};
  }
  for (int device : devices) {
    if (device >= tcim::GetDeviceNum() || device < 0) {
      throw std::invalid_argument(
          "devices must be within the valid device range");
    }
  }

  device_ctc_check(devices, prefill_path, decode_path, visual_path);

  if (prefill_path.extension() != ".hmms" && devices.size() > 1) {
    throw std::invalid_argument(
        "Multi-device testing is only supported for .hmms model files.");
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
  std::cout << "input token len : " << format_int_list(input_token_lens)
            << std::endl;
  std::cout << "stop token len : " << format_int_list(stop_token_lens)
            << std::endl;
  std::cout << "devices : " << format_int_list(devices) << std::endl;
  std::cout << "loop : " << loop_round << std::endl;
  std::cout << "batch : " << batch << std::endl;
  uint32_t warm_up_input = 0;
  uint32_t warm_up_output = 0;
  if (warm_up_enable) {
    std::cout << "warm_up : enable" << std::endl;
    warm_up_input = args.count("warm_up_input")
                        ? validate_setting(args, "warm_up_input")
                        : input_token_lens.front();
    warm_up_output = args.count("warm_up_output")
                         ? validate_setting(args, "warm_up_output")
                         : stop_token_lens.front();
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
  for (size_t i = 0; i < input_token_lens.size(); ++i) {
    settings.perf_cases.push_back({input_token_lens[i], stop_token_lens[i]});
  }
  settings.input_tokens_len = input_token_lens.front();
  settings.stop_tokens_len = stop_token_lens.front();
  settings.devices = devices;
  settings.batch_size = batch;
  settings.LazyMode = lazy_mode_enable;
  settings.warm_up = warm_up_enable;
  settings.warm_up_input = warm_up_input;
  settings.warm_up_output = warm_up_output;
  settings.loop_count = loop_round;
  settings.skip_perf = skip_perf;
  settings.perf_case_total = static_cast<int>(settings.perf_cases.size());

  return settings;
}

int RunPerf(std::unordered_map<std::string, std::string> args) {
  int interval =
      args.count("interval") ? validate_setting(args, "interval") : 500;
  interval = std::min(std::max(interval, 300), 60000);
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

    std::unordered_map<int, DeviceStats> init_dev_stats =
        device_monitor->getDeviceStats();
    std::unique_ptr<HmllmInferBase> Qwen3Infer;
    if (settings.visual_path.empty()) {
      if (settings.batch_size == 1) {
        Qwen3Infer = std::make_unique<HmllmInfer>(
            settings.prefill_path, settings.decode_path,
            settings.embedding_path, settings.devices, settings.batch_size,
            settings.LazyMode);
      } else {
        if (houmo_target != "xh2") {
          throw std::runtime_error(
              "Only xh2 support multibacth, device not match!");
        }
        Qwen3Infer = std::make_unique<HmllmInferMultiBatch>(
            settings.prefill_path, settings.decode_path,
            settings.embedding_path, settings.devices, settings.batch_size,
            settings.LazyMode);
      }
    } else {
      Qwen3Infer = std::make_unique<HmvllmInfer>(
          settings.prefill_path, settings.decode_path, settings.embedding_path,
          settings.visual_path, settings.devices, settings.batch_size,
          settings.LazyMode);
    }
    std::unordered_map<int, DeviceStats> post_init_dev_stats =
        device_monitor->getDeviceStats();
#if defined(__linux__)
    host_mem_info = host_mem_monitor->getCurrentMemoryInfo();
#endif
    if (!settings.skip_perf) {
      auto perf_tracker = Qwen3Infer->get_perf_tracker();
      if (settings.warm_up) {
        std::cout << "\n"
                  << std::string(30, '=') << "(v)LLM Perf WarmUp: input "
                  << settings.warm_up_input << ", output "
                  << settings.warm_up_output << std::string(30, '=') << "\n ";
        float current_temperature = device_monitor->getCurrentTemperature();
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
        perf_tracker->reset();
        Qwen3Infer->perf_llm(settings.warm_up_input, settings.warm_up_output);
        perf_tracker->pref_delete_warmup();
        std::cout << "\n" << std::string(82, '=') << "\n";
#if defined(__linux__)
        max_host_mem_info = host_mem_monitor->getMaxMemoryInfo();
#endif
        std::unordered_map<int, DeviceStats> current_dev_stats =
            device_monitor->getDeviceStats();
        InferenceMetricsWithLoadTime perf_metrics =
            perf_tracker->get_perf_current_summary();
        perf_dumper.writePerfBrief(settings, perf_metrics, host_mem_info,
                                   max_host_mem_info, post_init_dev_stats,
                                   current_dev_stats, "llm-perf warmup");
      }

      for (size_t perf_case_index = 0;
           perf_case_index < settings.perf_cases.size(); ++perf_case_index) {
        PerfSettings current_settings = BuildPerfCaseSettings(
            settings, settings.perf_cases[perf_case_index], perf_case_index);
        perf_tracker->pref_delete_warmup();

        std::cout << COLOR_BLUE << "\n"
                  << std::string(24, '=')
                  << "(v)LLM Perf Case: " << current_settings.perf_case_index
                  << "/" << current_settings.perf_case_total << " | input "
                  << current_settings.input_tokens_len << " | output "
                  << current_settings.stop_tokens_len << std::string(24, '=')
                  << "\n ";

        for (int i = 0; i < settings.loop_count; ++i) {
          std::cout << COLOR_BLUE << "\n"
                    << std::string(24, '=')
                    << "(v)LLM Perf Loop Progress: " << (i + 1) << "/"
                    << settings.loop_count << " | case "
                    << current_settings.perf_case_index << "/"
                    << current_settings.perf_case_total << std::string(24, '=')
                    << "\n ";
          float current_temperature = device_monitor->getCurrentTemperature();
          std::cout << "Device temperature: " << current_temperature << " °C"
                    << std::endl;
          if (current_temperature > ALARM_TEMPERATURE_THRESHOLD &&
              current_temperature < SHUTDOWN_TEMPERATURE_THRESHOLD) {
            std::cout << COLOR_YELLOW
                      << "Device temperature is beyond 80.0 °C, Temperature "
                         "Warning!"
                      << COLOR_RESET << std::endl;
          }
          if (current_temperature >= SHUTDOWN_TEMPERATURE_THRESHOLD) {
            throw std::runtime_error(
                "Device temperature is beyond 100.0 °C, "
                "Shutdown the demo!");
          }
          perf_tracker->reset();
          Qwen3Infer->perf_llm(current_settings.input_tokens_len,
                               current_settings.stop_tokens_len);
          std::cout << "\n" << std::string(82, '=') << "\n";
#if defined(__linux__)
          max_host_mem_info = host_mem_monitor->getMaxMemoryInfo();
#endif
          std::unordered_map<int, DeviceStats> current_dev_stats =
              device_monitor->getDeviceStats();
          InferenceMetricsWithLoadTime current_perf_metrics =
              perf_tracker->get_perf_current_summary();
          perf_dumper.writePerfBrief(
              current_settings, current_perf_metrics, host_mem_info,
              max_host_mem_info, post_init_dev_stats, current_dev_stats,
              "llm-perf Loop Progress: " + std::to_string(i + 1) + "/" +
                  std::to_string(settings.loop_count));
        }

        perf_tracker->showSummary(true);
#if defined(__linux__)
        max_host_mem_info = host_mem_monitor->getMaxMemoryInfo();
#endif
        std::unordered_map<int, DeviceStats> current_dev_stats =
            device_monitor->getDeviceStats();
        InferenceMetricsWithLoadTime perf_metrics =
            perf_tracker->get_perf_avg_summary();
        perf_dumper.dumpPerf(current_settings, perf_metrics, host_mem_info,
                             max_host_mem_info, post_init_dev_stats,
                             current_dev_stats);
        perf_dumper.showPerfBrief(current_settings, perf_metrics, host_mem_info,
                                  max_host_mem_info, post_init_dev_stats,
                                  current_dev_stats);
        perf_dumper.writePerfBrief(
            current_settings, perf_metrics, host_mem_info, max_host_mem_info,
            post_init_dev_stats, current_dev_stats,
            "llm-perf Case Average: " +
                std::to_string(current_settings.perf_case_index) + "/" +
                std::to_string(current_settings.perf_case_total));
      }
      std::cout << COLOR_RESET;
    }

    InferenceMetricsWithLoadTime skip_perf_metrics;
    if (settings.skip_perf) {
      skip_perf_metrics =
          Qwen3Infer->get_perf_tracker()->get_perf_avg_summary();
    }

    Qwen3Infer.reset();
    device_monitor->stop();
    std::unordered_map<int, DeviceStats> end_dev_stats =
        device_monitor->getFinalDeviceStats();

#if defined(__linux__)
    host_mem_monitor->stop();
    max_host_mem_info = host_mem_monitor->getFinalMemoryInfo();
#endif
    if (settings.skip_perf) {
      perf_dumper.dumpPerf(settings, skip_perf_metrics, host_mem_info,
                           max_host_mem_info, post_init_dev_stats,
                           end_dev_stats);
      perf_dumper.showPerfBrief(settings, skip_perf_metrics, host_mem_info,
                                max_host_mem_info, post_init_dev_stats,
                                end_dev_stats);
      perf_dumper.writePerfBrief(settings, skip_perf_metrics, host_mem_info,
                                 max_host_mem_info, post_init_dev_stats,
                                 end_dev_stats, "llm-perf Average");
    }
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