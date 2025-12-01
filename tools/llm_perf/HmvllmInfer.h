#ifndef __VLLMINFER_H__
#define __VLLMINFER_H__

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <regex>
#include <sstream>

#include "HmEmbedding.h"
#include "tcim/tcim_runtime.h"
#include "utils.h"

class HmvllmInfer : public HmllmInferBase {
 public:
  HmvllmInfer(const std::string &prefillModelPath,
              const std::string &decodeModelPath,
              const std::string &embeddingWeightPath,
              const std::string &vitModelPath, int ndevices, int batches);
  HmvllmInfer(const HmvllmInfer &it) = delete;
  HmvllmInfer &operator=(const HmvllmInfer &it) = delete;
  HmvllmInfer(HmvllmInfer &&it) noexcept = default;
  HmvllmInfer &operator=(HmvllmInfer &&it) noexcept = default;
  ~HmvllmInfer();

  void DebugModelInfo(tcim::Module &module, const std::string &modelName);
  PerfInfos perf_llm(const uint32_t input_tokens_len,
                     const uint32_t stop_tokens_len) override;

 private:
  std::string prefillModelPath = "";
  std::string decodeModelPath = "";
  std::string vitModelPath = "";

  int prefill_length = 0;
  int embedding_length = 0;
  int context_max_length = 0;
  int batch = 0;
  int argmax_dim_len = 0;

  std::shared_ptr<HmEmbedding> embedding;

  tcim::Module::WeightManager weight_manager;
  std::shared_ptr<tcim::Module> prefill_module;
  std::shared_ptr<tcim::Module> decode_module;
  std::shared_ptr<tcim::Module> vit_module;

  std::vector<std::string> dummy_names;

  std::map<std::string, tcim::Tensor> prefill_input_map;
  std::map<std::string, tcim::Tensor> decode_input_map;
  std::map<std::string, tcim::Tensor> vit_input_map;

  std::map<std::string, tcim::Tensor> prefill_output_map;
  std::map<std::string, tcim::Tensor> decode_output_map;
  std::map<std::string, tcim::Tensor> vit_output_map;
  int bar_width = 50;
  int attn_idx_start = 0;
  int vit_input_nums = 0;

  std::vector<char *> prefill_input_ptrs;
  std::vector<char *> decode_input_ptrs;
  std::vector<char *> vit_input_ptrs;

 private:
  int get_nblocks();
  int get_attn_idx_start();

  void PrefillSetInputDatas(void *data);
  float PrefillInfer();
  void PrefillGetOutputDatas(std::vector<int32_t> &ids);

  void DecodeSetInputDatas(void *data);
  float DecodeInfer();
  void DecodeGetOutputDatas(std::vector<int32_t> &ids);

  void VitSetInput();
  float VitInfer();
  void VitGetOutputDatas();
};

#endif  // __VLLMINFER_H__