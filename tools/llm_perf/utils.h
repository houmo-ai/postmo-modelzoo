#ifndef __UTILS_H__
#define __UTILS_H__

#include <codecvt>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <locale>
#include <nlohmann/json.hpp>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using json = nlohmann::json;

typedef enum {
  PERFCMD = 0,
  PERFJSON,
  PERFINVAILD
} PerfConfigType;

void HelpUsage(char* argv[]) {
  std::cout << "Usage: " << argv[0]
            << " --key value [options...]\n\n"
               "Options:\n"
               "  -c, --config    FILE      use json file to start llm_perf, "
               "cat template config.json for more message\n"
               "Or:\n"
               "  --prefill       FILE      prefill model file\n"
               "  --decode        FILE      decode model file\n"
               "  --embedding     FILE      embedding weight file\n"
               "  --input         NUM       number of input tokens\n"
               "  --stop          NUM       number of tokens to generate\n"
               "  --ndevices      NUM       device count\n"
               "  --loop          NUM       loop test rounds\n"
               "  -h, --help                show help message\n";
}

std::unordered_map<std::string, std::string> parse_json(const json &j) {
  std::unordered_map<std::string, std::string> args;
  for (auto& [key, val] : j.items()) {
    args[key] = val.is_string() ? val.get<std::string>() : val.dump();
  }

  return args;
}

PerfConfigType ParsePerfRunType(int argc, char* argv[]) {
  if(argc == 1) {
    return PerfConfigType::PERFINVAILD;
  }

  if (argc == 2) {
    std::string arg = argv[1];
    if (arg == "-h" || arg == "--help") {
      return PerfConfigType::PERFINVAILD;
    }
  }

  if (argc == 3) {
    const std::string arg = argv[1];
    if (arg == "-c" || arg == "--config") {
      return PerfConfigType::PERFJSON;
    } 
  }

  return PerfConfigType::PERFCMD;
}

/**
 * 解析命令行参数，支持 --key value 格式
 * @param argc 命令行参数数量
 * @param argv 命令行参数数组
 * @return 解析后的参数映射（key: 参数名, value: 参数值）
 */
std::unordered_map<std::string, std::string> parse_args(int argc,
                                                        char* argv[]) {
  std::unordered_map<std::string, std::string> args;

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[1];
    if (arg == "-c" || arg == "--config" || arg == "-h" || arg == "--help") {
      std::cerr << "Invalid args!" << std::endl;
      HelpUsage(argv);
      std::exit(0);
    }
  }

  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    if (arg.substr(0, 2) == "--") {
      std::string key = arg.substr(2);
      if (i + 1 >= argc) {
        throw std::invalid_argument("Missing value for argument: " + arg);
      }
      std::string value = argv[++i];
      args[key] = value;
    } else {
      throw std::invalid_argument("Invalid argument format: " + arg +
                                  " (use --key value)");
    }
  }

  return args;
}

/**
 * 验证路径是否存在（文件或目录）
 * @param path 路径字符串
 * @param arg_name 参数名（用于错误提示）
 * @return 标准化后的路径
 */
fs::path validate_path(std::unordered_map<std::string, std::string>& args,
                       const std::string& arg_name) {
  fs::path path;
  if (args.find(arg_name) != args.end()) {
    if (args[arg_name].empty()) {
      throw std::invalid_argument("Missing " + arg_name + " value (use --" +
                                  arg_name + " <value>).");
    }
    std::string path_str = args[arg_name];
    path = fs::u8path(path_str);  // 支持 Unicode 路径（跨平台）
    // path = fs::canonical(fs::absolute(path));

    if (!fs::exists(path)) {
      throw std::invalid_argument(arg_name +
                                  " path does not exist: " + path.u8string());
    }
  } else {
    throw std::invalid_argument("Missing arg : " + arg_name + ", (use --" +
                                arg_name + " to set arg).");
  }
  return path;
}

int validate_setting(std::unordered_map<std::string, std::string>& args,
                     const std::string& arg_name) {
  int value;
  if (args.find(arg_name) != args.end()) {
    if (args[arg_name].empty()) {
      throw std::invalid_argument("Missing " + arg_name + " value (use --" +
                                  arg_name + " <value>).");
    }

    value = stoi(args[arg_name]);
    if (value <= 0) {
      throw std::invalid_argument("Invalid " + arg_name + " value (use --" +
                                  arg_name + " <value> to set valid value).");
    }
  } else {
    throw std::invalid_argument("Missing arg : " + arg_name + ", (use --" +
                                arg_name + " to set arg).");
  }
  return value;
}

#endif  // __UTILS_H__