/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmvllmInfer.cc
 * Description:
 *   HmvllmInfer Implementation - Performance testing implementation for
 * vision-language large language model inference.
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
#include "HmvllmInfer.h"

HmvllmInfer::HmvllmInfer(const std::string &prefillModelPath,
                         const std::string &decodeModelPath,
                         const std::string &embeddingWeightPath,
                         const std::string &visionModelPath,
                         const std::vector<int> &devices, int batches,
                         bool LazyMode) {
  perf_tracker = std::make_shared<InferencePerformanceTracker>();
  this->prefillModelPath = prefillModelPath;
  this->decodeModelPath = decodeModelPath;
  this->visionModelPath = visionModelPath;
  // create device manager and weight manager
  tcim::DevManager dev_manager = tcim::DevManager::Create(devices);
  weight_manager =
      tcim::Module::WeightManager::CreateWeightManager(dev_manager);
  // create option
  auto option_prefill = tcim::Module::Option(weight_manager);
  auto option_decode = tcim::Module::Option(weight_manager);
  auto option_vision = tcim::Module::Option(weight_manager);
  option_prefill.EnableHostLazyLoading(true);
  option_decode.EnableHostLazyLoading(true);
  option_vision.EnableHostLazyLoading(true);
  if (LazyMode) {
    option_prefill.EnableIOLazyMode(true);
    option_decode.EnableIOLazyMode(true);
    option_vision.EnableIOLazyMode(true);
  } else {
    option_prefill.EnableIOLazyMode(false);
    option_decode.EnableIOLazyMode(false);
    option_vision.EnableIOLazyMode(false);
  }
  // init module
  prefill_module = std::make_shared<tcim::Module>();
  perf_tracker->perfStart(PerfType::PREFILL_LOAD_TIME);
  CHECK_TCIM_RET_STATUS(
      prefill_module->LoadModel(prefillModelPath, option_prefill));
  perf_tracker->perfEnd(PerfType::PREFILL_LOAD_TIME);

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

  decode_module = std::make_shared<tcim::Module>();
  perf_tracker->perfStart(PerfType::DECODE_LOAD_TIME);
  CHECK_TCIM_RET_STATUS(
      decode_module->LoadModel(decodeModelPath, option_decode));
  perf_tracker->perfEnd(PerfType::DECODE_LOAD_TIME);

  vision_module = std::make_shared<tcim::Module>();
  perf_tracker->perfStart(PerfType::VISION_LOAD_TIME);
  CHECK_TCIM_RET_STATUS(
      vision_module->LoadModel(visionModelPath, option_vision));
  perf_tracker->perfEnd(PerfType::VISION_LOAD_TIME);

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

  vision_input_nums = vision_module->GetInputNum();
  prefill_input_init();
  decode_input_init();
  vision_input_init();

  // Configure additional inputs for decode module (KV cache inputs)
  size_t total_mem_size = 0;
  for (int idx = 0; idx < this->prefill_module->GetInputNum(); idx++) {
    const std::string layer_name = prefill_module->GetInputName(idx);
    if (layer_name.find("model_layers") != std::string::npos) {
      auto cache = prefill_module->GetDevInput(layer_name);
      CHECK_TCIM_RET_STATUS(decode_module->SetDevInput(layer_name, cache));
      auto input_info = prefill_module->GetInputInfo(layer_name);
      total_mem_size += input_info.MemSize();
    }

    if (layer_name.find("conv_cache") != std::string::npos) {
      std::string output_name = layer_name;
      const std::string prefix = "past_conv_cache_";
      if (output_name.rfind(prefix, 0) == 0) {
        output_name.replace(0, prefix.size(), "conv_cache_out_");
      }
      auto cache = prefill_module->GetDevInput(layer_name);
      CHECK_TCIM_RET_STATUS(prefill_module->SetDevOutput(output_name, cache));
      CHECK_TCIM_RET_STATUS(decode_module->SetDevInput(layer_name, cache));
      CHECK_TCIM_RET_STATUS(decode_module->SetDevOutput(output_name, cache));
      auto input_info = prefill_module->GetInputInfo(layer_name);
      total_mem_size += input_info.MemSize();
    }

    if (layer_name.find("recurrent_state") != std::string::npos) {
      std::string output_name = layer_name;
      const std::string prefix = "past_recurrent_state_";
      if (output_name.rfind(prefix, 0) == 0) {
        output_name.replace(0, prefix.size(), "recurrent_state_out_");
      }
      auto cache = prefill_module->GetDevInput(layer_name);
      CHECK_TCIM_RET_STATUS(prefill_module->SetDevOutput(output_name, cache));
      CHECK_TCIM_RET_STATUS(decode_module->SetDevInput(layer_name, cache));
      CHECK_TCIM_RET_STATUS(decode_module->SetDevOutput(output_name, cache));
      auto input_info = prefill_module->GetInputInfo(layer_name);
      total_mem_size += input_info.MemSize();
    }
  }
  double kvcache_mem_size =
      static_cast<double>(total_mem_size) / (1024.0 * 1024.0);
  perf_tracker->set_kvcache_mem(kvcache_mem_size);

  embedding = std::make_shared<HmEmbedding>(
      embeddingWeightPath, this->embedding_length, this->prefill_length);
  vocab_size = embedding->get_vocab_size();
}

void HmvllmInfer::prefill_input_init() {
  if (prefill_module == nullptr) {
    return;
  }
  prefill_input_map.clear();
  for (int idx = 0; idx < attn_idx_start; ++idx) {
    auto input_name = prefill_module->GetInputName(idx);
    auto input_info = prefill_module->GetInputInfo(input_name).AsContiguous();
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    size_t memSize = input_tensor.MemSize();
    prefill_input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    if (input_name.find("position_ids") != std::string::npos ||
        input_name.find("_embed") != std::string::npos) {
      prefill_input_datas.insert(
          std::pair<std::string, std::unique_ptr<char[]>>(
              input_name, std::make_unique<char[]>(memSize)));
      std::fill(prefill_input_datas.at(input_name).get(),
                prefill_input_datas.at(input_name).get() + memSize, char(0));
    }
  }

  for (int idx = attn_idx_start; idx < prefill_module->GetInputNum(); idx++) {
    auto input_name = prefill_module->GetInputName(idx);
    if (input_name.find("conv_cache") != std::string::npos ||
        input_name.find("recurrent_state") != std::string::npos) {
      auto input_info = prefill_module->GetInputInfo(input_name).AsContiguous();
      tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
      prefill_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
      memset(input_tensor.Buffer().Data(), 0, input_tensor.MemSize());
      CHECK_TCIM_RET_STATUS(prefill_module->SetInput(input_name, input_tensor));
      CHECK_TCIM_RET_STATUS(decode_module->SetInput(input_name, input_tensor));
    }
  }
}

void HmvllmInfer::decode_input_init() {
  if (decode_module == nullptr) {
    return;
  }
  decode_input_map.clear();
  for (int idx = 0; idx < attn_idx_start; ++idx) {
    auto input_name = decode_module->GetInputName(idx);
    auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    size_t memSize = input_info.MemSize();
    decode_input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    if (input_name.find("position_ids") != std::string::npos ||
        input_name.find("_embed") != std::string::npos) {
      decode_input_datas.insert(std::pair<std::string, std::unique_ptr<char[]>>(
          input_name, std::make_unique<char[]>(memSize)));
      std::fill(decode_input_datas.at(input_name).get(),
                decode_input_datas.at(input_name).get() + memSize, char(0));
    }
  }
}

void HmvllmInfer::vision_input_init() {
  if (vision_module == nullptr) {
    return;
  }
  vision_input_map.clear();
  vision_input_datas.clear();
  for (int idx = 0; idx < vision_input_nums; ++idx) {
    auto input_name = vision_module->GetInputName(idx);
    auto input_info = vision_module->GetInputInfo(input_name).AsContiguous();
    size_t memSize = input_info.MemSize();
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    vision_input_map.insert(
        std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    vision_input_datas.insert(std::pair<std::string, std::unique_ptr<char[]>>(
        input_name, std::make_unique<char[]>(memSize)));
  }
}

void HmvllmInfer::DebugModelInfo(tcim::Module &module,
                                 const std::string &modelName) {
  std::cout << std::string(50, '=') << " ModelInfo of " << modelName << " "
            << std::string(50, '=') << std::endl;
  int input_num = module.GetInputNum();
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();

    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
  }

  int output_num = module.GetOutputNum();
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
  }
}

int HmvllmInfer::get_attn_idx_start() {
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

int HmvllmInfer::get_nblocks() {
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

HmvllmInfer::~HmvllmInfer() {
  prefill_module.reset();
  decode_module.reset();
  vision_module.reset();
}

void HmvllmInfer::PrefillSetInputDatas(void *data, int current_length) {
  for (int idx = 0; idx < attn_idx_start; idx++) {
    auto name = prefill_module->GetInputName(idx);
    auto tensor = prefill_input_map.at(name);
    size_t memSize = tensor.MemSize();

    if (name.find("input_1") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data, memSize));
    } else if (name.find("position_ids") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(
          prefill_input_datas.at(name).get(), memSize));
    } else if (name.find("valid_length") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&past_seq_len, memSize));
    } else if (name.find("current_length") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&current_length, memSize));
    } else if (name.find("_embed") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(
          prefill_input_datas.at(name).get(), memSize));
    } else if (name.find("attn_mask") != std::string::npos) {
      std::vector<tensor_type> attn_mask;
      for (int i = 0; i < this->prefill_length; ++i) {
        if (i < current_length) {
          attn_mask.emplace_back(static_cast<tensor_type>(1.0f));
        } else {
          attn_mask.emplace_back(static_cast<tensor_type>(0.0f));
        }
      }
    } else {
      continue;
    }

    perf_tracker->perfStart(PerfType::PREFILL_INPUT_TIME);
    CHECK_TCIM_RET_STATUS(prefill_module->SetInput(name, tensor));
    perf_tracker->perfEnd(PerfType::PREFILL_INPUT_TIME);
  }

  DebugSetInputValue(prefill_module, 4, 6);
}

void HmvllmInfer::PrefillInfer() {
  DebugSetInputValue(prefill_module, 4, 6);
  CHECK_TCIM_RET_STATUS(prefill_module->Run());
  DebugSetInputValue(prefill_module, 4, 6);
  CHECK_TCIM_RET_STATUS(prefill_module->Sync());
  DebugSetInputValue(prefill_module, 4, 6);
  return;
}

void HmvllmInfer::PrefillGetOutputDatas(std::vector<int32_t> &ids) {
  int output_num = prefill_module->GetOutputNum();

  auto output_name = prefill_module->GetOutputName(0);
  auto output_info = prefill_module->GetOutputInfo(output_name).AsContiguous();
  perf_tracker->perfStart(PerfType::PREFILL_OUTPUT_TIME);
  auto dev_output_tensor = prefill_module->GetDevOutput(output_name);
  auto host_output_tensor = dev_output_tensor.ToHost(true);
  perf_tracker->perfEnd(PerfType::PREFILL_OUTPUT_TIME);

  void *prefill_outData = host_output_tensor.Buffer().Data();
  ids.emplace_back(eigen_argmax<tensor_type>(
      static_cast<tensor_type *>(prefill_outData), argmax_dim_len));
}

void HmvllmInfer::DecodeSetInputDatas(void *data, int valid_length) {
  for (int idx = 0; idx < attn_idx_start; idx++) {
    auto name = decode_module->GetInputName(idx);
    auto tensor = decode_input_map.at(name);
    size_t memSize = tensor.MemSize();
    if (name.find("input_1") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(data, memSize));
    } else if (name.find("valid_length") != std::string::npos) {
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&valid_length, memSize));
    } else if (name.find("current_length") != std::string::npos) {
      int decode_current_length = 1;
      CHECK_TCIM_RET_STATUS(
          tensor.Buffer().CopyFromHost(&decode_current_length, memSize));
    } else if (name.find("attn_mask") != std::string::npos) {
      tensor_type mask_value = static_cast<tensor_type>(1.0f);
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(&mask_value, memSize));
    } else {
      CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(
          decode_input_datas.at(name).get(), memSize));
    }
    perf_tracker->perfStart(PerfType::DECODE_INPUT_TIME);
    CHECK_TCIM_RET_STATUS(decode_module->SetInput(name, tensor));
    perf_tracker->perfEnd(PerfType::DECODE_INPUT_TIME);
  }

  DebugSetInputValue(decode_module, 4, 6);
}

void HmvllmInfer::DecodeInfer() {
  DebugSetInputValue(decode_module, 4, 6);
  CHECK_TCIM_RET_STATUS(decode_module->Run());
  DebugSetInputValue(decode_module, 4, 6);
  CHECK_TCIM_RET_STATUS(decode_module->Sync());
  DebugSetInputValue(decode_module, 4, 6);
  return;
}

void HmvllmInfer::DecodeGetOutputDatas(std::vector<int32_t> &ids) {
  int output_num = decode_module->GetOutputNum();
  auto output_name = decode_module->GetOutputName(0);
  auto output_info = decode_module->GetOutputInfo(output_name).AsContiguous();

  perf_tracker->perfStart(PerfType::DECODE_OUTPUT_TIME);
  auto dev_output_tensor = decode_module->GetDevOutput(output_name);
  auto host_output_tensor = dev_output_tensor.ToHost(true);
  perf_tracker->perfEnd(PerfType::DECODE_OUTPUT_TIME);

  void *decode_outData = host_output_tensor.Buffer().Data();
  ids.emplace_back(eigen_argmax<tensor_type>(
      static_cast<tensor_type *>(decode_outData), argmax_dim_len));
}

void HmvllmInfer::VisionSetInput() {
  for (int idx = 0; idx < vision_input_nums; idx++) {
    auto input_name = vision_module->GetInputName(idx);
    auto tensor = vision_input_map.at(input_name);
    size_t memSize = tensor.MemSize();
    CHECK_TCIM_RET_STATUS(tensor.Buffer().CopyFromHost(
        vision_input_datas.at(input_name).get(), memSize));
    perf_tracker->perfStart(PerfType::VISION_INPUT_TIME);
    CHECK_TCIM_RET_STATUS(vision_module->SetInput(input_name, tensor));
    perf_tracker->perfEnd(PerfType::VISION_INPUT_TIME);
  }
}

void HmvllmInfer::VisionInfer() {
  CHECK_TCIM_RET_STATUS(vision_module->Run());
  CHECK_TCIM_RET_STATUS(vision_module->Sync());
  return;
}

void HmvllmInfer::VisionGetOutputDatas() {
  int output_num = vision_module->GetOutputNum();
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = vision_module->GetOutputName(idx);
    perf_tracker->perfStart(PerfType::VISION_OUTPUT_TIME);
    auto dev_output_tensor = vision_module->GetDevOutput(output_name);
    // make sure the host tensor is contiguous
    auto host_output_tensor = dev_output_tensor.ToHost(true);
    perf_tracker->perfEnd(PerfType::VISION_OUTPUT_TIME);
  }
}

void HmvllmInfer::perf_llm(const uint32_t input_tokens_len,
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

  // 1. prepare inputs
  std::vector<int> all_input_ids =
      generateRandomVector(input_tokens_len, vocab_size);
  PerfInfos vllm_perf_datas;
  memset(&vllm_perf_datas, 0, sizeof(PerfInfos));

  perf_tracker->perfStart(PerfType::VISION_TOTAL_TIME);
  VisionSetInput();
  perf_tracker->perfStart(PerfType::VISION_INFER_TIME);
  VisionInfer();
  perf_tracker->perfEnd(PerfType::VISION_INFER_TIME);
  VisionGetOutputDatas();
  perf_tracker->perfEnd(PerfType::VISION_TOTAL_TIME);

  vllm_perf_datas.input_tokens = input_tokens_len;
  if (input_tokens_len + stop_tokens_len > context_max_length) {
    std::cout << "input_tokens_len + stop_tokens_len > context_max_length, "
                 "cast stop_tokens_len to "
              << context_max_length - input_tokens_len << std::endl;
    vllm_perf_datas.stop_tokens = context_max_length - input_tokens_len;
  } else {
    vllm_perf_datas.stop_tokens = stop_tokens_len;
  }

  perf_tracker->perfStart(PerfType::PREFILL_TOTAL_TIME);
  int prefill_loop_round =
      std::ceil((float)input_tokens_len / (float)prefill_length);
  valid_length = 0, current_length = 0;
  past_seq_len = 0;
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

    PrefillSetInputDatas(input_datas, current_length);
    past_seq_len += current_length;
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
  ids.clear();
  PrefillGetOutputDatas(ids);

  perf_tracker->perfEnd(PerfType::PREFILL_TOTAL_TIME);
  int context_length = input_tokens_len;

  do {
    if ((context_length > context_max_length) ||
        (vllm_perf_datas.decode_count >= vllm_perf_datas.stop_tokens)) {
      break;
    }
    perf_tracker->perfStart(PerfType::DECODE_TOTAL_TIME);
    perf_tracker->perfStart(PerfType::DECODE_EMBED_TIME);
    input_datas = embedding->EmbeddingTokens(ids);
    perf_tracker->perfEnd(PerfType::DECODE_EMBED_TIME);

    DecodeSetInputDatas(static_cast<void *>(input_datas), context_length);

    perf_tracker->perfStart(PerfType::DECODE_INFER_TIME);
    DecodeInfer();
    perf_tracker->perfEnd(PerfType::DECODE_INFER_TIME);
    ids.clear();

    DecodeGetOutputDatas(ids);
    vllm_perf_datas.decode_count++;
    perf_tracker->perfEnd(PerfType::DECODE_TOTAL_TIME);
    double ratio = static_cast<double>(vllm_perf_datas.decode_count) /
                   vllm_perf_datas.stop_tokens;
    int filled = static_cast<int>(ratio * bar_width);
    std::cout << '\r' << "Decode: " << std::setw(3) << int(ratio * 100)
              << "% | " << std::string(filled, '*')
              << std::string(bar_width - filled, ' ') << "| "
              << vllm_perf_datas.decode_count << '/'
              << vllm_perf_datas.stop_tokens << std::flush;

    context_length++;
  } while (true);

  perf_tracker->setBasicInfo(1, input_tokens_len, vllm_perf_datas.stop_tokens,
                             1);
  // perf information
  perf_tracker->showSummary();
  return;
}