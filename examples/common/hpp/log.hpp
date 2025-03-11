// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file log.hpp
 */

#ifndef __LOG_HPP__
#define __LOG_HPP__

#include <iostream>
#include <sstream>
#include <string>


#define COLOR_RED     "\x1b[91;20m"
#define COLOR_GREEN   "\x1b[92;20m"
#define COLOR_YELLOW  "\x1b[93;20m"
#define COLOR_BLUE    "\x1b[94;20m"
#define COLOR_MAGENT  "\x1b[95;20m"
#define COLOR_CYAN    "\x1b[96;20m"
#define COLOR_RESET   "\x1b[0m"


class Logger {
 public:
  Logger(bool log_enable = true) : log_enable_(log_enable) {}
  ~Logger() {
    if (log_enable_) std::cout << COLOR_RESET << std::endl;
  }

  template <typename T>
  inline Logger& operator<<(T t) {
    if (log_enable_) std::cout << t;
    return *this;
  }

  template <typename T>
  inline Logger& operator<<(const std::vector<T>& vec) {
    std::cout << "[";
    for (size_t i = 0; i < vec.size(); ++i) {
      std::cout << vec[i];
      if (i != vec.size() - 1) {
        std::cout << ", ";
      }
    }
    std::cout << "]";
    return *this;
  }

 private:
  bool log_enable_ = true;
};

#define LOG_SUCCESS Logger() << COLOR_GREEN << "[success] "
#define LOG_ERROR   Logger() << COLOR_RED << "[error] "
#define LOG_WARN    Logger() << COLOR_YELLOW << "[warn] "
#define LOG_INFO    Logger() << COLOR_CYAN << "[info] "
#define LOG_DEBUG   Logger() << COLOR_MAGENT << "[debug] "


#endif // __LOG_HPP__
