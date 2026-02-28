/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmllmInfer.cc
 * Description:
 *   HmllmInfer Implementation - Performance testing implementation for large
 * language model inference.
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
#include "HmllmInfer.h"

HmllmInfer::HmllmInfer(const std::string &prefillModelPath,
                       const std::string &decodeModelPath,
                       const std::string &embeddingWeightPath, int ndevices,
                       int batches, bool LazyMode) {
  perf_tracker = std::make_shared<InferencePerformanceTracker>();
  this->prefillModelPath = prefillModelPath;
  this->decodeModelPath = decodeModelPath;
  // Create weightManager
  std::vector<int> devs;
  devs.clear();
  std::cout << "Use Devices ";
  for (int i = 0; i < ndevices; i++) {
    devs.emplace_back(i);
    std::cout << i << " ";
  }

  std::cout << std::endl;
  // Create device manager with the specified devices
  tcim::DevManager dev_manager = tcim::DevManager::Create(devs);
  // Create weight manager using the device manager
  weight_manager =
      tcim::Module::WeightManager::CreateWeightManager(dev_manager);
  // Create weightManager options for prefill and decode modules
  auto option_prefill = tcim::Module::Option(weight_manager);
  auto option_decode = tcim::Module::Option(weight_manager);
  // Enable lazy mode
  if (LazyMode) {
    option_prefill.EnableIOLazyMode(true);
    option_decode.EnableIOLazyMode(true);
  } else {
    option_prefill.EnableIOLazyMode(false);
    option_decode.EnableIOLazyMode(false);
  }
  // Initialize prefill module and load the model
  prefill_module = std::make_shared<tcim::Module>();
  perf_tracker->perfStart(PerfType::PREFILL_LOAD_TIME);
  CHECK_TCIM_RET_STATUS(
      prefill_module->LoadModel(prefillModelPath, option_prefill));
  perf_tracker->perfEnd(PerfType::PREFILL_LOAD_TIME);
  // Get number of blocks in the model and create dummy names for cache inputs
  int n_blocks = get_nblocks();
  for (int i = 0; i < n_blocks; i++) {
    std::stringstream ss;
    ss << "model_layers_" << i << "_self_attn_kcache_input";
    dummy_names.emplace_back(ss.str());
  }

  for (int i = 0; i < n_blocks; i++) {
    std::stringstream ss;
    ss << "model_layers_" << i << "_self_attn_vcache_input";
    dummy_names.emplace_back(ss.str());
  }

  // Set dummy tensors for the decode module
  option_decode.SetDummyTensors(dummy_names);
  decode_module = std::make_shared<tcim::Module>();
  perf_tracker->perfStart(PerfType::DECODE_LOAD_TIME);
  CHECK_TCIM_RET_STATUS(
      decode_module->LoadModel(decodeModelPath, option_decode));
  perf_tracker->perfEnd(PerfType::DECODE_LOAD_TIME);

  // Get model configuration parameters
  attn_idx_start = get_attn_idx_start();
  this->prefill_length =
      prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[1];
  this->embedding_length =
      prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[2];
  this->context_max_length =
      prefill_module->GetInputInfo(prefill_module->GetInputName(attn_idx_start))
          .Shape()[2];
  this->batch =
      decode_module->GetInputInfo(decode_module->GetInputName(0)).Shape()[0];
  this->argmax_dim_len =
      decode_module->GetOutputInfo(decode_module->GetOutputName(0)).Shape()[2];

  if (this->batch != batches) {
    throw std::runtime_error("Model Batch Not match args batch!");
  }

  // Configure decode module's other inputs (KV cache)
  for (int idx = attn_idx_start; idx < 2 * n_blocks + attn_idx_start; idx++) {
    const std::string kvcache_name = prefill_module->GetInputName(idx);
    auto kvcache = prefill_module->GetDevInput(kvcache_name);
    CHECK_TCIM_RET_STATUS(decode_module->SetDevInput(kvcache_name, kvcache));
  }

  // Initialize embedding module with weights path, embedding length and prefill
  // length
  embedding = std::make_shared<HmEmbedding>(
      embeddingWeightPath, this->embedding_length, this->prefill_length);

  prefill_input_init();
  decode_input_init();

  perf_tracker->reset();
}

void HmllmInfer::prefill_input_init() {
  if (prefill_module == nullptr) {
    return;
  }
  prefill_input_map.clear();
  for (int idx = 0; idx < attn_idx_start; ++idx) {
    auto input_name = prefill_module->GetInputName(idx);
    auto input_info = prefill_module->GetInputInfo(input_name).AsContiguous();
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    prefill_input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
  }
}

void HmllmInfer::decode_input_init() {
  if (decode_module == nullptr) {
    return;
  }
  decode_input_map.clear();
  for (int idx = 0; idx < attn_idx_start; ++idx) {
    auto input_name = decode_module->GetInputName(idx);
    auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    decode_input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
  }
}

int HmllmInfer::get_attn_idx_start() {
  int start = 0;
  static const std::regex pattern(
      R"(^model_layers_(\d+)_self_attn_kcache_input$)");
  int input_num = prefill_module->GetInputNum();
  for (int idx = 0; idx < input_num; idx++) {
    std::string input_name = prefill_module->GetInputName(idx);
    if (std::regex_match(input_name, pattern)) {
      start = idx;
      break;
    }
  }
  return start;
}

int HmllmInfer::get_nblocks() {
  int count = 0;
  static const std::regex pattern(
      R"(^model_layers_(\d+)_self_attn_kcache_input$)");
  int input_num = prefill_module->GetInputNum();
  for (int idx = 0; idx < input_num; idx++) {
    std::string input_name = prefill_module->GetInputName(idx);
    if (std::regex_match(input_name, pattern)) {
      ++count;
    }
  }
  return count;
}

void HmllmInfer::DebugModelInfo(tcim::Module &module,
                                const std::string &modelName) {
  std::cout << std::string(50, '=') << " ModelInfo of " << modelName << " "
            << std::string(50, '=') << std::endl;
  int input_num = module.GetInputNum();
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    size_t memSize = input_info.MemSize();
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
  }

  int output_num = module.GetOutputNum();
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
  }
}

HmllmInfer::~HmllmInfer() {
  prefill_module.reset();
  decode_module.reset();
  embedding.reset();
}

void HmllmInfer::PrefillSetInputDatas(void *data, int32_t valid_length,
                                      int32_t current_length) {
  for (int idx = 0; idx < attn_idx_start; idx++) {
    auto input_name = prefill_module->GetInputName(idx);
    auto tensor = prefill_input_map.at(input_name);
    size_t memSize = tensor.MemSize();
    if (idx == 0) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data, memSize));
    } else if (idx == 1) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&valid_length, memSize));
    } else if (idx == 2) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&current_length, memSize));
    } else if (idx == 3) {
      std::vector<int32_t> position_ids;
      for (int i = valid_length; i < valid_length + this->prefill_length; ++i) {
        position_ids.emplace_back(i);
      }
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(position_ids.data(), memSize));
    } else {
      continue;
    }
    perf_tracker->perfStart(PerfType::PREFILL_INPUT_TIME);
    CHECK_TCIM_RET_STATUS(prefill_module->SetInput(input_name, tensor));
    perf_tracker->perfEnd(PerfType::PREFILL_INPUT_TIME);
  }

  DebugSetInputValue(prefill_module, 1, attn_idx_start);
}

void HmllmInfer::PrefillInfer() {
  DebugSetInputValue(prefill_module, 1, attn_idx_start);
  CHECK_TCIM_RET_STATUS(prefill_module->Run());
  DebugSetInputValue(prefill_module, 1, attn_idx_start);
  CHECK_TCIM_RET_STATUS(prefill_module->Sync());
  DebugSetInputValue(prefill_module, 1, attn_idx_start);
  return;
}

void HmllmInfer::PrefillGetOutputDatas(std::vector<int32_t> &ids) {
  int output_num = prefill_module->GetOutputNum();
  auto output_name = prefill_module->GetOutputName(0);
  auto output_info = prefill_module->GetOutputInfo(output_name).AsContiguous();

  perf_tracker->perfStart(PerfType::PREFILL_OUTPUT_TIME);
  auto dev_output_tensor = prefill_module->GetDevOutput(output_name);
  perf_tracker->perfEnd(PerfType::PREFILL_OUTPUT_TIME);
  auto host_output_tensor = dev_output_tensor.ToHost(true);

  void *prefill_outData = host_output_tensor.Buffer().Data();
  ids.emplace_back(eigen_argmax<tensor_type>(
      static_cast<tensor_type *>(prefill_outData), argmax_dim_len));
}

void HmllmInfer::DecodeSetInputDatas(void *data, int32_t context_length) {
  for (int idx = 0; idx < attn_idx_start; idx++) {
    auto input_name = decode_module->GetInputName(idx);
    auto tensor = decode_input_map.at(input_name);
    size_t memSize = tensor.MemSize();
    if (idx == 0) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data, memSize));
    } else if (idx == 1) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&context_length, memSize));
    } else if (idx == 2) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&decode_current_length, memSize));
    } else if (idx == 3) {
      int32_t position_id = context_length + 1;
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&position_id, memSize));
    } else {
      continue;
    }
    perf_tracker->perfStart(PerfType::DECODE_INPUT_TIME);
    CHECK_TCIM_RET_STATUS(decode_module->SetInput(input_name, tensor));
    perf_tracker->perfEnd(PerfType::DECODE_INPUT_TIME);
  }

  DebugSetInputValue(decode_module, 1, attn_idx_start);
}

void HmllmInfer::DecodeInfer() {
  DebugSetInputValue(decode_module, 1, attn_idx_start);
  CHECK_TCIM_RET_STATUS(decode_module->Run());
  DebugSetInputValue(decode_module, 1, attn_idx_start);
  CHECK_TCIM_RET_STATUS(decode_module->Sync());
  DebugSetInputValue(decode_module, 1, attn_idx_start);
  return;
}

void HmllmInfer::DecodeGetOutputDatas(std::vector<int32_t> &ids) {
  int output_num = decode_module->GetOutputNum();
  auto output_name = decode_module->GetOutputName(0);
  auto output_info = decode_module->GetOutputInfo(output_name).AsContiguous();

  perf_tracker->perfStart(PerfType::DECODE_OUTPUT_TIME);
  auto dev_output_tensor = decode_module->GetDevOutput(output_name);
  perf_tracker->perfEnd(PerfType::DECODE_OUTPUT_TIME);
  auto host_output_tensor = dev_output_tensor.ToHost(true);

  void *decode_outData = host_output_tensor.Buffer().Data();
  ids.emplace_back(eigen_argmax<tensor_type>(
      static_cast<tensor_type *>(decode_outData), argmax_dim_len));
}

void HmllmInfer::perf_llm(const uint32_t input_tokens_len,
                          const uint32_t stop_tokens_len) {
  if (input_tokens_len > context_max_length) {
    throw std::runtime_error("Question long than " +
                             std::to_string(context_max_length) +
                             ", please shorten it !");
  }

  int32_t valid_length = 0, current_length = this->prefill_length;
  tensor_type *input_datas = nullptr;
  std::vector<int> input_ids;
  std::vector<int> ids;
  PerfInfos llm_perf_datas;
  memset(&llm_perf_datas, 0, sizeof(PerfInfos));
  // 1. prepare inputs
  std::vector<int> all_input_ids = generateRandomVector(input_tokens_len);
  if (input_tokens_len + stop_tokens_len > context_max_length) {
    std::cout << "input_tokens_len + stop_tokens_len > context_max_length, "
                 "cast stop_tokens_len to "
              << context_max_length - input_tokens_len << std::endl;
    llm_perf_datas.stop_tokens = context_max_length - input_tokens_len;
  } else {
    llm_perf_datas.stop_tokens = stop_tokens_len;
  }

  perf_tracker->perfStart(PerfType::PREFILL_TOTAL_TIME);
  int prefill_loop_round =
      std::ceil((float)input_tokens_len / (float)prefill_length);
  valid_length = 0, current_length = 0;

  for (int round = 0; round < prefill_loop_round; round++) {
    valid_length = round * prefill_length;
    std::vector<int> input_ids;
    if (round == prefill_loop_round - 1) {
      current_length = input_tokens_len - round * prefill_length;
      input_ids.reserve(current_length);
      input_ids.assign(all_input_ids.end() - current_length,
                       all_input_ids.end());
    } else {
      current_length = prefill_length;
      input_ids.reserve(current_length);
      input_ids.assign(all_input_ids.begin() + round * prefill_length,
                       all_input_ids.begin() + (round + 1) * prefill_length);
    }

    perf_tracker->perfStart(PerfType::PREFILL_EMBED_TIME);
    input_datas = embedding->EmbeddingTokens(input_ids);
    perf_tracker->perfEnd(PerfType::PREFILL_EMBED_TIME);

    PrefillSetInputDatas(input_datas, valid_length, current_length);

    perf_tracker->perfStart(PerfType::PREFILL_INFER_TIME);
    PrefillInfer();
    perf_tracker->perfEnd(PerfType::PREFILL_INFER_TIME);
    float prefill_ratio = (float)(round + 1) / (float)prefill_loop_round;
    int filled = static_cast<int>(prefill_ratio * bar_width);
    std::cout << '\r' << "Prefill: " << std::setw(3) << int(prefill_ratio * 100)
              << "% |" << std::string(filled, '*')
              << std::string(bar_width - filled, ' ') << "| " << std::flush;
  }
  std::cout << std::endl;
  PrefillGetOutputDatas(ids);
  perf_tracker->perfEnd(PerfType::PREFILL_TOTAL_TIME);

  int context_length = input_tokens_len;
  do {
    if ((context_length > context_max_length) ||
        (llm_perf_datas.decode_count >= llm_perf_datas.stop_tokens)) {
      break;
    }
    perf_tracker->perfStart(PerfType::DECODE_TOTAL_TIME);
    perf_tracker->perfStart(PerfType::DECODE_EMBED_TIME);
    input_datas = embedding->EmbeddingTokens(ids);
    perf_tracker->perfEnd(PerfType::DECODE_EMBED_TIME);

    DecodeSetInputDatas(static_cast<void *>(input_datas),
                        static_cast<int32_t>(context_length));

    perf_tracker->perfStart(PerfType::DECODE_INFER_TIME);
    DecodeInfer();
    perf_tracker->perfEnd(PerfType::DECODE_INFER_TIME);
    ids.clear();
    DecodeGetOutputDatas(ids);

    llm_perf_datas.decode_count++;
    perf_tracker->perfEnd(PerfType::DECODE_TOTAL_TIME);
    double ratio = static_cast<double>(llm_perf_datas.decode_count) /
                   llm_perf_datas.stop_tokens;
    int filled = static_cast<int>(ratio * bar_width);
    std::cout << '\r' << "Decode: " << std::setw(3) << int(ratio * 100) << "% |"
              << std::string(filled, '*')
              << std::string(bar_width - filled, ' ') << "| "
              << llm_perf_datas.decode_count << '/'
              << llm_perf_datas.stop_tokens << std::flush;

    context_length++;
  } while (true);
  perf_tracker->setBasicInfo(1, input_tokens_len, llm_perf_datas.decode_count,
                             0);
  // perf information
  perf_tracker->showSummary();
  return;
}