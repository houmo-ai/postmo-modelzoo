#pragma once
#include <string>

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

// 打印堆栈信息（跨平台）
static void print_stacktrace() {
#ifdef _WIN32
  // Windows 堆栈信息捕获
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
  // Linux/MacOS 堆栈信息捕获
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

// 定义 LOG_ASSERT 宏，检查条件并打印堆栈信息
#define LOG_ASSERT(condition)                        \
  do {                                               \
    if (!(condition)) {                              \
      LOG_ERROR("Assertion failed: {}", #condition); \
      print_stacktrace();                            \
      LOG_FLUSH;                                     \
      std::abort();                                  \
    }                                                \
  } while (0)

static void InitGoogleLogging() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  std::vector<spdlog::sink_ptr> sinks{console_sink};
  auto logger = std::make_shared<spdlog::logger>(CONSOLE_LOGGER, sinks.begin(),
                                                 sinks.end());
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);
  spdlog::set_pattern("%^%L%Y%m%d %T.%6f %t %s:%#] %v%$");
  // LOG_INFO("SPDLOG Version: {}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR,
  //          SPDLOG_VER_PATCH);
}

static void InitGoogleLogging(const std::string &logpath,
                              size_t rotating_file_size,
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
  auto logger =
      std::make_shared<spdlog::logger>(logger_name, sinks.begin(), sinks.end());
  spdlog::register_logger(logger);
  spdlog::set_default_logger(logger);
  spdlog::set_pattern("%^%L%Y%m%d %T.%6f %t %s:%#] %v%$");
  // LOG_INFO("SPDLOG Version: {}.{}.{}", SPDLOG_VER_MAJOR, SPDLOG_VER_MINOR,
  //          SPDLOG_VER_PATCH);
}