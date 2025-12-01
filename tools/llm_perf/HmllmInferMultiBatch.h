#ifndef __HMLLMINFERMULTIBATCH_H__
#define __HMLLMINFERMULTIBATCH_H__

#include <chrono>
#include <iomanip>
#include <map>
#include <random>
#include <regex>
#include <sstream>

#include "HmEmbedding.h"
#include "tcim/tcim_runtime.h"

struct PerfSingleBatchInfo {
  uint32_t input_tokens;
  float time;
  float embedding_time;
  std::vector<int> next_id;
};

class HmllmInferMultiBatch : public HmllmInferBase {
 public:
  HmllmInferMultiBatch(const std::string &prefillModelPath,
                       const std::string &decodeModelPath,
                       const std::string &embeddingWeightPath, int ndevices,
                       int batches);
  HmllmInferMultiBatch(const HmllmInferMultiBatch &it) = delete;
  HmllmInferMultiBatch &operator=(const HmllmInferMultiBatch &it) = delete;
  HmllmInferMultiBatch(HmllmInferMultiBatch &&it) noexcept = default;
  HmllmInferMultiBatch &operator=(HmllmInferMultiBatch &&it) noexcept = default;
  ~HmllmInferMultiBatch();

  void DebugModelInfo(tcim::Module &module, const std::string &modelName);

  PerfInfos perf_llm(const uint32_t input_tokens_len,
                     const uint32_t stop_tokens_len) override;

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

  std::shared_ptr<HmEmbedding> embedding;

  tcim::Module::WeightManager weight_manager;
  std::shared_ptr<tcim::Module> prefill_module;
  std::shared_ptr<tcim::Module> decode_module;

  std::vector<std::string> dummy_names;

  std::map<std::string, tcim::Tensor> prefill_input_map;
  std::map<std::string, tcim::Tensor> decode_input_map;
  std::map<std::string, tcim::Tensor> prefill_output_map;
  std::map<std::string, tcim::Tensor> decode_output_map;

  std::vector<std::vector<int>> next_ids;
  std::vector<int> current_echo_lens;

  int bar_width = 50;
  int n_blocks = 0;

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
  void PrefillSetInputDatas(void *data, int32_t valid_length,
                            int32_t current_length);
  // perfill模型推理
  void PrefillInfer();
  // 获取prefill输出的token ids
  void PrefillGetOutputDatas(std::vector<int32_t> &ids);

  PerfSingleBatchInfo run_prefill(int batch,
                                  const std::vector<int> all_input_ids);
  PerfSingleBatchInfo run_decode(tensor_type *input_datas,
                                 const std::vector<int> context_length);
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
  void DecodeGetOutputDatas();
};

#endif  // __HMLLMINFERMULTIBATCH_H__