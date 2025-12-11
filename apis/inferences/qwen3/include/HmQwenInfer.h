#ifndef __HMQWENINFER_H__
#define __HMQWENINFER_H__

#include <iomanip>
#include <map>
#include <regex>

#include "Hmtokenizer.h"
#include "tcim/tcim_runtime.h"
#include "utils.h"

/**
 * 存储perf结果
 */
struct PerfInfos {
  int input_tokens;
  int output_tokens;
  float prefill_time;
  float decode_time;
  float embedding_time;
};

/**
  class HmQwenInfer
 */
class HmQwenInfer {
 public:
  HmQwenInfer(const std::string &prefillModelPath,
              const std::string &decodeModelPath,
              const std::string &tokenizerJsonPath,
              const std::string &embeddingWeightPath);
  HmQwenInfer(const HmQwenInfer &it) = delete;
  HmQwenInfer &operator=(const HmQwenInfer &it) = delete;
  HmQwenInfer(HmQwenInfer &&it) noexcept = default;
  HmQwenInfer &operator=(HmQwenInfer &&it) noexcept = default;
  ~HmQwenInfer();

  /**
   * 获取模型Module
   * @param model_type        可选参数 0-prefill 1-decode others-返回nullptr
   * @return                  对应模型Module管理的shared_ptr,入参错误返回nullptr
   */
  std::shared_ptr<tcim::Module> GetModule(int model_type);
  /**
   * 获取模型tokenizer
   * @return                  HmTokenizer的shared_ptr
   */
  std::shared_ptr<HmTokenizer> GetTokenizer();
  /**
   * 打印模型输入输出信息
   * @param module            模型Module
   * @param modelName         模型名称
   * @return                  无返回值，打印输入输出信息
   */
  void DebugModelInfo(tcim::Module &module, const std::string &modelName);
  /**
   * Qwen3 问答函数
   * @param msg               用户输入字符串
   * @return                  无返回值，打印大模型对话的问答信息及性能信息
   */
  void Chat(const std::string &msg);

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
  int32_t decode_current_length = 1;
  // model related
  std::shared_ptr<HmTokenizer> tokenizer;

  tcim::Module::WeightManager weight_manager;
  std::shared_ptr<tcim::Module> prefill_module;
  std::shared_ptr<tcim::Module> decode_module;

  std::vector<std::string> dummy_names;

  std::map<std::string, tcim::Tensor> prefill_input_map;
  std::map<std::string, tcim::Tensor> decode_input_map;
  std::map<std::string, tcim::Tensor> prefill_output_map;
  std::map<std::string, tcim::Tensor> decode_output_map;

 private:
  // 获取prefill的nblocks
  int GetnBlocks();

  /**
   * 设置prefill输入
   * @param data              prefill输入0
   * @param valid_length      prefill输入1
   * @param current_length    prefill输入2
   * @return                  无返回值，设置输入数据
   */
  void PrefillSetInputDatas(void *data, int32_t valid_length,
                            int32_t current_length);
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

#endif  // __HMQWENINFER_H__