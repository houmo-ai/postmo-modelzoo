#include <codecvt>
#include <filesystem>
#include <iostream>
#include <locale>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "HmllmInfer.h"
#include "HmllmInferMultiBatch.h"
#include "HmvllmInfer.h"
#include "tcim/tcim_runtime.h"
#include "utils.h"

#ifdef _MSC_VER
#include <Windows.h>
#endif

int RunPerf(std::unordered_map<std::string, std::string> args) {
  fs::path prefill_path = validate_path(args, "prefill");  // 工作目录
  fs::path decode_path = validate_path(args, "decode");    // 模型文件路径
  fs::path visual_path =
      args.count("visual") ? validate_path(args, "visual") : fs::path();
  fs::path embedding_path = validate_path(args, "embedding");  // 分词器路径
  int input_token_len = validate_setting(args, "input");
  int stop_token_len = validate_setting(args, "stop");
  int ndevices =
      args.count("ndevices") ? validate_setting(args, "ndevices") : 1;
  int loop_round = args.count("loop") ? validate_setting(args, "loop") : 1;

  int batch = args.count("batch") ? validate_setting(args, "batch") : 1;
  bool warm_up_enable = args.count("no_warm_up") ? false : true;

  std::cout << COLOR_YELLOW << std::string(25, '=') << " Perf Settings "
            << std::string(25, '=') << std::endl;
  std::cout << "prefill path : " << prefill_path.string() << std::endl;
  std::cout << "decode path : " << decode_path.string() << std::endl;
  if (!visual_path.empty()) {
    std::cout << "visual path : " << visual_path.string() << std::endl;
  }
  std::cout << "embedding path : " << embedding_path.string() << std::endl;
  std::cout << "input token len : " << input_token_len << std::endl;
  std::cout << "stop token len : " << stop_token_len << std::endl;
  std::cout << "ndevices : " << ndevices << std::endl;
  std::cout << "loop : " << loop_round << std::endl;
  std::cout << "batch : " << batch << std::endl;
  if (warm_up_enable) {
    std::cout << "warm_up : enable" << std::endl;
  } else {
    std::cout << "warm_up : disable" << std::endl;
  }
  std::cout << std::string(65, '=') << COLOR_RESET << std::endl;

  const char* houmo_target_env = getenv("HOUMO_TARGET");
  std::string houmo_target =
      houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
  if (houmo_target != "xh2" && houmo_target != "xh1") {
    throw std::invalid_argument("Unsupported backend " + houmo_target);
  }

  if (houmo_target == "xh1") {
    if (batch != 1) {
      throw std::runtime_error("xh1 only support single-bacth !");
    }
  }

  std::unique_ptr<HmllmInferBase> Qwen3Infer;
  if (visual_path.empty()) {
    if (batch == 1) {
      Qwen3Infer = std::make_unique<HmllmInfer>(
          prefill_path.string(), decode_path.string(), embedding_path.string(),
          ndevices, batch);
    } else {
      if (houmo_target != "xh2") {
        throw std::runtime_error(
            "Only xh2 support multibacth, device not match!");
      }
      Qwen3Infer = std::make_unique<HmllmInferMultiBatch>(
          prefill_path.string(), decode_path.string(), embedding_path.string(),
          ndevices, batch);
    }
  } else {
    Qwen3Infer = std::make_unique<HmvllmInfer>(
        prefill_path.string(), decode_path.string(), embedding_path.string(),
        visual_path.string(), ndevices, batch);
  }

  PerfInfos avg_perfdata, total_perfdata;
  memset(&avg_perfdata, 0, sizeof(PerfInfos));
  memset(&total_perfdata, 0, sizeof(PerfInfos));
  if (warm_up_enable) {
    int32_t warm_up_len = 256;
    std::cout << "\n"
              << std::string(30, '=') << "(v)LLM Perf WarmUp: input "
              << warm_up_len << ", output " << warm_up_len
              << std::string(30, '=') << "\n ";
    Qwen3Infer->perf_llm(warm_up_len, warm_up_len);
    std::cout << std::string(82, '=') << "\n";
  }

  for (int i = 0; i < loop_round; ++i) {
    std::cout << COLOR_BLUE << "\n"
              << std::string(30, '=')
              << "(v)LLM Perf Loop Progress: " << (i + 1) << "/" << loop_round
              << std::string(30, '=') << "\n ";
    PerfInfos perf_data = Qwen3Infer->perf_llm(input_token_len, stop_token_len);
    std::cout << std::string(82, '=') << "\n";
    total_perfdata.input_tokens = perf_data.input_tokens;
    total_perfdata.stop_tokens = perf_data.stop_tokens;
    total_perfdata.prefill_time += perf_data.prefill_time;
    total_perfdata.decode_time += perf_data.decode_time;
    total_perfdata.embedding_time += perf_data.embedding_time;
    total_perfdata.t_total += perf_data.t_total;
    total_perfdata.ttft += perf_data.ttft;
    total_perfdata.decode_count += perf_data.decode_count;
  }

  avg_perfdata.input_tokens = total_perfdata.input_tokens;
  avg_perfdata.stop_tokens = total_perfdata.stop_tokens;
  avg_perfdata.prefill_time = total_perfdata.prefill_time / loop_round;
  avg_perfdata.decode_time = total_perfdata.decode_time / loop_round;
  avg_perfdata.embedding_time = total_perfdata.embedding_time / loop_round;
  avg_perfdata.t_total = total_perfdata.t_total / loop_round;
  avg_perfdata.decode_count = total_perfdata.decode_count / loop_round;
  avg_perfdata.ttft = total_perfdata.ttft / loop_round;
  std::cout << COLOR_GREEN << std::string(30, '=')
            << " (v)LLM Perf Avarage Information " << std::string(30, '=')
            << "\n";
  ShowPerfInformation(avg_perfdata);
  std::cout << COLOR_GREEN << std::string(90, '=') << "\n";
  std::cout << COLOR_RESET;
  Qwen3Infer.reset();
  return 0;
}

int RunPerfJson(int argc, char* argv[]) {
  const std::string jsonfile = argv[2];
  fs::path path = fs::u8path(jsonfile);
  if (!fs::exists(path)) {
    throw std::invalid_argument("config path does not exist: " +
                                path.u8string());
  }

  std::ifstream f(jsonfile);
  json perf_configs;
  f >> perf_configs;
  int n_tasks = perf_configs["Streams"].size();
  int curTaskId = 0;
  for (json& stream : perf_configs["Streams"]) {
    std::cout << COLOR_GREEN << std::string(45, '#') << "Start of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ", ModelName:" << stream["ModelName"] << "."
              << std::string(45, '#') << "\n";

    std::unordered_map<std::string, std::string> args = parse_json(stream);
    RunPerf(args);
    std::cout << COLOR_GREEN << std::string(45, '#') << " End of Task "
              << (curTaskId + 1) << ", All Task:" << n_tasks
              << ",  ModelName:" << stream["ModelName"] << "."
              << std::string(45, '#') << "\n\n\n";
    curTaskId++;
  }
  std::cout << COLOR_RESET;
  return 0;
}

int main(int argc, char* argv[]) {
#ifdef _MSC_VER
  SetConsoleOutputCP(CP_UTF8);
  SetConsoleCP(CP_UTF8);
#endif
  try {
    // 1. 解析命令行参数
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
