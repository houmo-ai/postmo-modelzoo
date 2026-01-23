/*
 * Copyright (c) 2025 HOUMO AI
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

#include <algorithm>
#include <cctype>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <locale>
#include <memory>
#include <nlohmann/json.hpp>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

#include "../utils.h"
using write_json = nlohmann::ordered_json;

class PerfDumper {
 public:
  PerfDumper();

  void setJsonFile(const std::string &json_file, bool run_json_perf);
  PerfDumper(const PerfDumper &it) = delete;
  PerfDumper &operator=(const PerfDumper &it) = delete;
  PerfDumper(PerfDumper &&it) noexcept = default;
  PerfDumper &operator=(PerfDumper &&it) noexcept = default;

  void dumpPerf(const PerfSettings &perf_settings,
                const InferenceMetricsWithLoadTime &metrics);

  void generateJsonFile();

  ~PerfDumper();

 private:
  std::string dump_file = "";
  write_json root;
  write_json perf_metrics;
};

#endif  // __PERF_DUMPER_H__