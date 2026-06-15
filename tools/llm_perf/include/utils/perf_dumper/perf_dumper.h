/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: perf_dumper.h
 * Description:
 *   perf_dumper Header File - Defines the PerfDumper class for perf results.
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
#ifndef __PERF_DUMPER_H__
#define __PERF_DUMPER_H__

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "utils/device_monitor/device_monitor.h"
#include "utils/host_monitor/host_monitor.h"
#include "utils/utils.h"

class PerfDumper {
 public:
  PerfDumper();

  void setYamlFile(const std::string &yaml_file, bool run_yaml_perf);
  PerfDumper(const PerfDumper &it) = delete;
  PerfDumper &operator=(const PerfDumper &it) = delete;
  PerfDumper(PerfDumper &&it) noexcept = default;
  PerfDumper &operator=(PerfDumper &&it) noexcept = default;

  void dumpPerf(const PerfSettings &perf_settings,
                const InferenceMetricsWithLoadTime &results,
                const HostMemoryInfo &host_mem_info,
                const HostMemoryInfo &max_host_mem_info,
                const std::unordered_map<int, DeviceStats> &post_init_dev_stats,
                const std::unordered_map<int, DeviceStats> &end_device_stats);

  void showPerfBrief(
      const PerfSettings &perf_settings,
      const InferenceMetricsWithLoadTime &results,
      const HostMemoryInfo &host_mem_info,
      const HostMemoryInfo &max_host_mem_info,
      const std::unordered_map<int, DeviceStats> &post_init_dev_stats,
      const std::unordered_map<int, DeviceStats> &end_device_stats);

  void writePerfBrief(
      const PerfSettings &perf_settings,
      const InferenceMetricsWithLoadTime &results,
      const HostMemoryInfo &host_mem_info,
      const HostMemoryInfo &max_host_mem_info,
      const std::unordered_map<int, DeviceStats> &post_init_dev_stats,
      const std::unordered_map<int, DeviceStats> &end_device_stats,
      std::string perf_intruduction);

  void generateYamlFile();

  ~PerfDumper();

 private:
  std::string dump_file = "";
  YAML::Node root;
  bool init_yaml = true;
  std::string log_file = "perf_dumper.log";
};

#endif  // __PERF_DUMPER_H__