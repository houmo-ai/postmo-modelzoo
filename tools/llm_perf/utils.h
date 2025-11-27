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
#include <memory>
#include <cctype>
#include <algorithm>
#include <regex>
#include <iomanip>
#include <random>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <chrono>

#define TOKEN_ID_MAX 150000

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

namespace fs = std::filesystem;
using json = nlohmann::json;

typedef enum {
  PERFCMD = 0,
  PERFJSON,
  PERFINVAILD
} PerfConfigType;

static void HelpUsage(char* argv[]) {
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
               "  --batch         NUM       if multibatch model only xh2 support!\n"
               "  --no_warm_up              disable warm up!\n"
               "  -h, --help                show help message\n";
}

static std::unordered_map<std::string, std::string> parse_json(const json &j) {
  std::unordered_map<std::string, std::string> args;
  for (auto& [key, val] : j.items()) {
    args[key] = val.is_string() ? val.get<std::string>() : val.dump();
  }

  return args;
}

static PerfConfigType ParsePerfRunType(int argc, char* argv[]) {
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
static std::unordered_map<std::string, std::string> parse_args(int argc,
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
static fs::path validate_path(std::unordered_map<std::string, std::string>& args,
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

static int validate_setting(std::unordered_map<std::string, std::string>& args,
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


struct PerfInfos
{
  uint32_t input_tokens;
  uint32_t stop_tokens;
  float prefill_time;
  float decode_time;
  float embedding_time;      
  float t_total;            //E2E Latency
  uint32_t decode_count;
};



static void ShowPerfInformation(PerfInfos llm_perf_datas){
  std::ostringstream os;
  os << "\n-------------------  Performance Summary  --------------------\n";
  os << std::left << std::setfill(' ');
  os << std::setw(30) << "Metric" << std::setw(30) << "Value" << '\n';
  os << std::string(62, '-') << '\n';

  auto token = [&](const std::string& name, auto val, const std::string& unit = "") {
      os << std::setw(30) << name << std::setw(30) << val << unit << '\n';
  };

  auto fmt = [](auto v, int prec, const char* unit) -> std::string {
    std::ostringstream o;
    o << std::fixed << std::setprecision(prec) << v << unit;
    return o.str();
  };

  token("Prefill Time",  fmt(llm_perf_datas.prefill_time, 2, " ms"));
  token("Decode Time",  fmt(llm_perf_datas.decode_time,  2, " ms"));
  token("Prefill Speed",fmt(llm_perf_datas.input_tokens / (llm_perf_datas.prefill_time * 0.001f), 2, " tokens/s"));
  token("Decode Speed", fmt(llm_perf_datas.decode_count / (llm_perf_datas.decode_time * 0.001f), 2, " tokens/s"));
  token("TTFT",         fmt(llm_perf_datas.prefill_time, 2, " ms"));
  token("TPOT",         fmt(llm_perf_datas.decode_time / llm_perf_datas.decode_count, 2, " ms/token"));
  token("E2E Latency",  fmt(llm_perf_datas.t_total * 0.001f, 2, " seconds"));
  token("E2E TPS",      fmt(llm_perf_datas.decode_count / (llm_perf_datas.t_total * 0.001f), 2, " tokens/s"));
  token("Embedding Time",fmt(llm_perf_datas.embedding_time, 2, " ms"));
  os << "--------------------------------------------------------------\n";
  std::cout << os.str();
}

static std::vector<int> generateRandomVector(int len)
{
  std::vector<int> result;
  if (len <= 0)
  {
    return result; // 处理无效长度（返回空向量）
  }

  // 使用当前时间作为随机种子，确保每次运行生成不同序列
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::mt19937 generator(seed); // 采用 Mersenne Twister 随机数引擎

  // 定义随机数范围：[0, 151642]
  std::uniform_int_distribution<int> distribution(0, TOKEN_ID_MAX);

  // 填充向量
  result.reserve(len); // 预分配内存，提高效率
  for (int i = 0; i < len; ++i)
  {
    result.push_back(distribution(generator));
  }

  return result;
}

/**
 * eigen矩阵库计算argmax
 * @param ptr          数组首地址
 * @param n                 数组元素个数
 * @return                  返回最大值索引
 */
template <typename T>
static int eigen_argmax(const T *ptr, std::size_t n) {
    using Eigen::Tensor;
    using Eigen::TensorMap;

    TensorMap<Tensor<const T, 1>> tm(static_cast<const T *>(ptr), n);

    Eigen::Tensor<Eigen::Index, 0> t = tm.argmax();
    Eigen::Index idx = t(0);

    return static_cast<int>(idx);
}


class HmllmInferBase {
public:
  HmllmInferBase() = default;
  virtual ~HmllmInferBase() = default;
  virtual PerfInfos perf_llm(const uint32_t input_tokens_len, const uint32_t stop_tokens_len) = 0; 
};

#endif  // __UTILS_H__