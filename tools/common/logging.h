/*
 * Copyright (c) 2022 HOUMO AI
 *
 * File: logging.h
 * Description:
 *   Logging Utilities Header File - Defines logging macros and SpdLogger class
 * for cross-platform logging with console and file output support.
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
#ifndef _APIS_COMMON_HPP_LOGGING_H_
#define _APIS_COMMON_HPP_LOGGING_H_

#include <string>

#define SPDLOG_ACTIVE_LEVEL SPDLOG_LEVEL_TRACE

#include "spdlog/pattern_formatter.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include "spdlog/sinks/stdout_sinks.h"
#include "spdlog/spdlog.h"
#include "spdlog/version.h"
#ifdef _WIN32
#include <dbghelp.h>
#include <windows.h>
#pragma comment(lib, "dbghelp.lib")
#else
#include <execinfo.h>
#endif

#define DEFAULT_LOGGER "default"  // console && file
#define CONSOLE_LOGGER "console"
#define FILE_LOGGER "file"

#define LOG_TRACE(...) SPDLOG_TRACE(__VA_ARGS__)
#define LOG_DEBUG(...) SPDLOG_DEBUG(__VA_ARGS__)
#define LOG_INFO(...) SPDLOG_INFO(__VA_ARGS__)
#define LOG_WARNING(...) SPDLOG_WARN(__VA_ARGS__)
#define LOG_ERROR(...) SPDLOG_ERROR(__VA_ARGS__)
#define LOG_FLUSH spdlog::default_logger_raw()->flush()

// Print stack trace information (cross-platform)
static void print_stacktrace() {
#ifdef _WIN32
  // Windows stack trace capture
  void *stack[100];
  HANDLE process = GetCurrentProcess();
  SymInitialize(process, NULL, TRUE);
  unsigned short frames = CaptureStackBackTrace(0, 100, stack, NULL);
  SYMBOL_INFO *symbol =
      (SYMBOL_INFO *)calloc(sizeof(SYMBOL_INFO) + 256 * sizeof(char), 1);
  symbol->MaxNameLen = 255;
  symbol->SizeOfStruct = sizeof(SYMBOL_INFO);

  LOG_ERROR("Stacktrace:");
  for (unsigned int i = 0; i < frames; ++i) {
    SymFromAddr(process, (DWORD64)(stack[i]), 0, symbol);
    LOG_ERROR("{}: {}", i, symbol->Name);
  }
  free(symbol);
#else
  // Linux/MacOS stack trace capture
  const int max_frames = 100;
  void *buffer[max_frames];
  int nptrs = backtrace(buffer, max_frames);
  char **symbols = backtrace_symbols(buffer, nptrs);

  if (!symbols) {
    LOG_ERROR("Failed to capture stack trace.");
    return;
  }

  LOG_ERROR("Stacktrace ({} frames):", nptrs);
  for (int i = 0; i < nptrs; ++i) {
    LOG_ERROR("  {}", symbols[i]);
  }

  free(symbols);
#endif
}

#define LOG_FATAL(...)      \
  do {                      \
    LOG_ERROR(__VA_ARGS__); \
    print_stacktrace();     \
    LOG_FLUSH;              \
    std::abort();           \
  } while (0)

// Define LOG_ASSERT macro to check condition and print stack trace
#define LOG_ASSERT(condition)                        \
  do {                                               \
    if (!(condition)) {                              \
      LOG_ERROR("Assertion failed: {}", #condition); \
      print_stacktrace();                            \
      LOG_FLUSH;                                     \
      std::abort();                                  \
    }                                                \
  } while (0)

class SpdLogger {
 public:
  SpdLogger() {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks{console_sink};
    auto logger = std::make_shared<spdlog::logger>(CONSOLE_LOGGER,
                                                   sinks.begin(), sinks.end());
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    auto log_level = get_spdlog_level();
    spdlog::set_level(log_level);
    spdlog::set_pattern("%^%L%Y%m%d %T.%6f %t %s:%#] %v%$");
  }

  SpdLogger(const std::string &logpath, size_t rotating_file_size,
            size_t rotating_file_num, bool alsologtostderr) {
    auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
    std::vector<spdlog::sink_ptr> sinks{console_sink};
    std::string logger_name = FILE_LOGGER;
    if (alsologtostderr) {
      auto file_sink = std::make_shared<spdlog::sinks::rotating_file_sink_mt>(
          logpath, rotating_file_size, rotating_file_num);
      sinks.emplace_back(file_sink);
      logger_name = DEFAULT_LOGGER;
    }
    auto logger = std::make_shared<spdlog::logger>(logger_name, sinks.begin(),
                                                   sinks.end());
    spdlog::register_logger(logger);
    spdlog::set_default_logger(logger);
    auto log_level = get_spdlog_level();
    spdlog::set_level(log_level);
    spdlog::set_pattern("%^%L%Y%m%d %T.%6f %t %s:%#] %v%$");
    // LOG_INFO("SPDLOG Version: {}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR,
    //          SPDLOG_VER_PATCH);
  }

  spdlog::level::level_enum get_spdlog_level() {
    // Read environment variable HM_SPDLOG_LEVEL
    const char *env_level = std::getenv("HM_SPDLOG_LEVEL");
    if (!env_level) {
      return spdlog::level::info;  // default level
    }

    std::string level_str = env_level;
    for (char &c : level_str) c = tolower(c);

    if (level_str == "trace") return spdlog::level::trace;
    if (level_str == "debug") return spdlog::level::debug;
    if (level_str == "info") return spdlog::level::info;
    if (level_str == "warn") return spdlog::level::warn;
    if (level_str == "error") return spdlog::level::err;
    if (level_str == "critical") return spdlog::level::critical;
    if (level_str == "off") return spdlog::level::off;

    return spdlog::level::info;
  }
};

template <typename T = int>
class AutoLoggerInitHelper {
 public:
  static SpdLogger logger;
};

template <typename T>
SpdLogger AutoLoggerInitHelper<T>::logger;

template class AutoLoggerInitHelper<int>;

#endif  // _APIS_COMMON_HPP_LOGGING_H_