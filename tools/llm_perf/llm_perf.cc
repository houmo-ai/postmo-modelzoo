/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: llm_perf.cc
 * Description:
 *   LLM Performance Testing Tool - Main application for perf.
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
#include "run_perf.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

/**
 * @brief Main entry point for the LLM Performance Testing Tool
 * This application runs performance tests on large language models with various
 * configurations
 *
 * @param argc Number of command line arguments
 * @param argv Array of command line argument strings
 * @return int Exit status code (0 for success, non-zero for errors)
 */
int main(int argc, char* argv[]) {
#ifdef _MSC_VER
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  try {
    PerfConfigType runtype = ParsePerfRunType(argc, argv);
    if (runtype == PerfConfigType::PERFCMD) {
      auto args = parse_args(argc, argv);
      return RunPerf(args);
    } else if (runtype == PerfConfigType::PERFJSON) {
      return RunPerfJson(argc, argv);
    } else {
      HelpUsage(argv);
      return -1;
    }
  } catch (const std::exception& e) {
    std::cerr << "Error: " << e.what() << std::endl;
    HelpUsage(argv);
    return 1;
  }

  return 0;
}
