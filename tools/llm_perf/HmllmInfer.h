#ifndef __LLMINFER_H__
#define __LLMINFER_H__

#include "tcim/tcim_runtime.h"
#include <map>
#include "HmEmbedding.h"
#include <regex>
#include <iomanip>
#include <sstream>
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


class HmllmInfer
{
public:
  HmllmInfer(const std::string &prefillModelPath,
              const std::string &decodeModelPath,
              const std::string &embeddingWeightPath,
              int ndevices);
  HmllmInfer(const HmllmInfer &it) = delete;
  HmllmInfer &operator=(const HmllmInfer &it) = delete;
  HmllmInfer(HmllmInfer &&it) noexcept = default;
  HmllmInfer &operator=(HmllmInfer &&it) noexcept = default;
  ~HmllmInfer();

  void DebugModelInfo(tcim::Module &module, const std::string &modelName);
  /**
   * Qwen3 问答函数
   * @param msg               用户输入字符串
   * @return                  无返回值，打印大模型对话的问答信息及性能信息
   */
  PerfInfos perf_llm(const uint32_t input_tokens_len, const uint32_t stop_tokens_len);

private:
  // 模型路径
  std::string prefillModelPath = "";
  std::string decodeModelPath = "";
  // 相关配置参数-> 模型读取
  int prefill_length = 0;
  int embedding_length = 0;
  int context_max_length = 0;
  int batch = 0;
  int eos_token_id = 0;
  int argmax_dim_len = 0;
#ifdef BACKEND_XH1
  int16_t decode_current_length = 1;
#else
  int32_t decode_current_length = 1;
#endif
  std::shared_ptr<HmEmbedding> embedding;

  tcim::Module::WeightManager weight_manager;
  std::shared_ptr<tcim::Module> prefill_module;
  std::shared_ptr<tcim::Module> decode_module;

  std::vector<std::string> dummy_names;

  std::map<std::string, tcim::Tensor> prefill_input_map;
  std::map<std::string, tcim::Tensor> decode_input_map;
  std::map<std::string, tcim::Tensor> prefill_output_map;
  std::map<std::string, tcim::Tensor> decode_output_map;

  int bar_width = 50; 

private:
  // 获取prefill的nblocks
  int get_nblocks();

  /**
   * 设置prefill输入
   * @param data              prefill输入0
   * @param valid_length      prefill输入1
   * @param current_length    prefill输入2
   * @return                  无返回值，设置输入数据
   */
  void PrefillSetInputDatas(void *data, int32_t valid_length, int32_t current_length);
  // perfill模型推理
  void PrefillInfer();
  // 获取prefill输出的token ids
  void PrefillGetOutputDatas(std::vector<int32_t> &ids);

  /**
   * 设置decode输入
   * @param data              decode输入0
   * @param context_length    decode输入1
   * @return                  无返回值，设置输入数据
   */
  void DecodeSetInputDatas(void *data, int32_t context_length);
  // decode模型推理
  void DecodeInfer();
  // 获取deocde输出的token ids
  void DecodeGetOutputDatas(std::vector<int32_t> &ids);
};

#endif // __LLMINFER_H__