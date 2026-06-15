/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: run_asr.h
 * Description:
 *   ASR Performance Test Runner - standalone from run_perf.h.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef RUN_ASR_H
#define RUN_ASR_H

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "asr/HmAsrInfer.h"
#include "utils/device_monitor/device_monitor.h"
#include "utils/utils.h"
#if defined(__linux__)
#include "utils/host_monitor/host_monitor.h"
#endif

namespace fs = std::filesystem;

static int RunAsrCore(const std::string& model_name,
                      const std::string& encode_path,
                      const std::string& prefill_path,
                      const std::string& decode_path,
                      const std::string& tokenizer_path,
                      const std::string& embedding_path,
                      const std::vector<std::string>& audio_paths,
                      const std::vector<int>& devices,
                      int interval) {
  AsrModelType asr_type =
      HmAsrInfer::DetectModelType(prefill_path, devices);

  std::cout << COLOR_YELLOW << std::string(25, '=') << " ASR Perf Settings "
            << std::string(25, '=') << "\n"
            << "model: " << model_name << " ("
            << HmAsrInfer::ModelTypeToString(asr_type) << ")\n"
            << "encode: " << encode_path << "\n"
            << "prefill: " << prefill_path << "\n"
            << "decode: " << decode_path << "\n"
            << "tokenizer: " << tokenizer_path << "\n";
  if (!embedding_path.empty())
    std::cout << "embedding: " << embedding_path << "\n";
  std::cout << "devices: " << format_int_list(devices) << "\n"
            << std::string(65, '=') << COLOR_RESET << std::endl;

  auto device_monitor = std::make_unique<DeviceMonitor>(interval);
  device_monitor->start();
#if defined(__linux__)
  auto host_mem_monitor = std::make_unique<HostMonitor>(interval);
  host_mem_monitor->start();
#endif

  auto asr = std::make_unique<HmAsrInfer>(
      asr_type, encode_path, prefill_path, decode_path,
      tokenizer_path, embedding_path, devices);

  std::cout << COLOR_YELLOW << "\nASR model load time: " << asr->GetLoadTimeMs()
            << " ms" << COLOR_RESET << std::endl;

  for (size_t ai = 0; ai < audio_paths.size(); ++ai) {
    std::cout << COLOR_BLUE << "\n"
              << std::string(24, '=') << " ASR Perf: audio "
              << (ai + 1) << "/" << audio_paths.size()
              << std::string(24, '=') << "\n";

    float t = device_monitor->getCurrentTemperature();
    std::cout << "Device temperature: " << t << " C" << std::endl;

    asr->Transcribe(audio_paths[ai]);

    std::cout << std::string(82, '=') << "\n";
  }

  asr.reset();
  device_monitor->stop();
#if defined(__linux__)
  host_mem_monitor->stop();
#endif

  return 0;
}

static int RunAsr(std::unordered_map<std::string, std::string> args) {
  try {
    fs::path encode_path = validate_path(args, "encode");
    fs::path prefill_path = validate_path(args, "prefill");
    fs::path decode_path = validate_path(args, "decode");
    fs::path tokenizer_path = validate_path(args, "tokenizer");

    std::string embedding_path;
    if (args.count("embedding")) {
      embedding_path = validate_path(args, "embedding").string();
    }

    std::string model_name;
    if (args.find("model_name") != args.end()) {
      model_name = args["model_name"];
    } else if (args.find("ModelName") != args.end()) {
      model_name = args["ModelName"];
    }

    std::vector<int> devices;
    if (args.count("devices")) {
      std::unordered_map<std::string, std::string> tmp;
      tmp["devices"] = args["devices"];
      devices = validate_multi_setting(tmp, "devices");
    } else if (getenv("HOUMO_VISIBLE_DEVICES") != nullptr) {
      std::unordered_map<std::string, std::string> tmp;
      tmp["devices"] = getenv("HOUMO_VISIBLE_DEVICES");
      devices = validate_multi_setting(tmp, "devices");
    } else {
      devices = {0};
    }

    std::vector<std::string> audio_paths;
    if (args.count("audio")) {
      std::stringstream ss(args["audio"]);
      std::string item;
      while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(),
                                  [](unsigned char c) { return std::isspace(c); }),
                   item.end());
        if (!item.empty()) {
          fs::path p = fs::u8path(item);
          if (!fs::exists(p))
            throw std::invalid_argument("audio path not found: " + item);
          audio_paths.push_back(p.string());
        }
      }
    } else {
      throw std::invalid_argument("Missing --audio");
    }

    int interval = args.count("interval") ? validate_setting(args, "interval") : 500;

    return RunAsrCore(model_name,
                      encode_path.string(), prefill_path.string(),
                      decode_path.string(), tokenizer_path.string(),
                      embedding_path, audio_paths, devices, interval);
  } catch (const std::exception& e) {
    std::cerr << "ASR Perf Error: " << e.what() << std::endl;
    return 1;
  }
}

static int RunAsrConfig(int argc, char* argv[]) {
  const std::string yamlfile = argv[2];
  fs::path path = fs::u8path(yamlfile);
  if (!fs::exists(path)) {
    throw std::invalid_argument("config path does not exist: " + path.u8string());
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

  for (const auto& stream : config["Streams"]) {
    std::string model_name = stream["ModelName"]
                                 ? stream["ModelName"].as<std::string>()
                                 : "unknown";

    std::cout << COLOR_GREEN << std::string(45, '#') << "Start of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ", ModelName:" << model_name << "."
              << std::string(45, '#') << "\n";

    std::string encode = stream["encode"].as<std::string>();
    std::string prefill = stream["prefill"].as<std::string>();
    std::string decode = stream["decode"].as<std::string>();
    std::string tokenizer = stream["tokenizer"].as<std::string>();
    std::string embedding =
        stream["embedding"] ? stream["embedding"].as<std::string>() : "";
    std::string audio = stream["audio"].as<std::string>();

    std::vector<int> devices;
    if (stream["devices"]) {
      std::string dev_str = stream["devices"].as<std::string>();
      std::unordered_map<std::string, std::string> tmp;
      tmp["devices"] = dev_str;
      devices = validate_multi_setting(tmp, "devices");
    } else {
      devices = {0};
    }

    std::vector<std::string> audio_paths;
    {
      std::stringstream ss(audio);
      std::string item;
      while (std::getline(ss, item, ',')) {
        item.erase(std::remove_if(item.begin(), item.end(),
                                  [](unsigned char c) { return std::isspace(c); }),
                   item.end());
        if (!item.empty()) {
          fs::path p = fs::u8path(item);
          if (!fs::exists(p))
            throw std::invalid_argument("audio path not found: " + item);
          audio_paths.push_back(p.string());
        }
      }
    }

    int interval = stream["interval"] ? stream["interval"].as<int>() : 500;

    RunAsrCore(model_name, encode, prefill, decode, tokenizer, embedding,
               audio_paths, devices, interval);

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
