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
                         const std::string &vitModelPath, int ndevices,
                         int batches) {
  this->prefillModelPath = prefillModelPath;
  this->decodeModelPath = decodeModelPath;
  this->vitModelPath = vitModelPath;
  // create weightManager
  std::vector<int> devs;
  devs.clear();
  std::cout << "Use Devices ";
  for (int i = 0; i < ndevices; i++) {
    devs.emplace_back(i);
    std::cout << i << " ";
  }
  std::cout << std::endl;
  tcim::DevManager dev_manager = tcim::DevManager::Create(devs);
  weight_manager =
      tcim::Module::WeightManager::CreateWeightManager(dev_manager);
  // create option
  auto option_prefill = tcim::Module::Option(weight_manager);
  auto option_decode = tcim::Module::Option(weight_manager);
  auto option_vit = tcim::Module::Option(weight_manager);
  option_prefill.EnableLazyMode(true);
  option_decode.EnableLazyMode(true);
  option_vit.EnableLazyMode(true);
  // init module
  prefill_module = std::make_shared<tcim::Module>();
  prefill_module->LoadModel(prefillModelPath, option_prefill);

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
  decode_module->LoadModel(decodeModelPath, option_decode);

  vit_module = std::make_shared<tcim::Module>();
  vit_module->LoadModel(vitModelPath, option_vit);

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
  // Configure additional inputs for decode module (KV cache inputs)
  for (int idx = attn_idx_start; idx < 2 * n_blocks + attn_idx_start; idx++) {
    const std::string input_name = prefill_module->GetInputName(idx);
    auto cache = prefill_module->GetInput(input_name);
    decode_input_map.insert(std::pair<std::string, tcim::Tensor>(
        decode_module->GetInputName(idx), cache));
  }

  embedding = std::make_shared<HmEmbedding>(
      embeddingWeightPath, this->embedding_length, this->prefill_length);

  prefill_input_ptrs.resize(attn_idx_start - 1);
  decode_input_ptrs.resize(attn_idx_start - 1);
  for (int i = 0; i < attn_idx_start - 1; ++i) {
    prefill_input_ptrs[i] = nullptr;
    decode_input_ptrs[i] = nullptr;
  }

  vit_input_nums = vit_module->GetInputNum();
  vit_input_ptrs.resize(vit_input_nums);
  for (int i = 0; i < vit_input_nums; ++i) {
    vit_input_ptrs[i] = nullptr;
  }

  // DebugModelInfo(*prefill_module.get(), prefillModelPath);
  // DebugModelInfo(*decode_module.get(), decodeModelPath);
  // DebugModelInfo(*vit_module.get(), vitModelPath);
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
  for (int i = 0; i < prefill_input_ptrs.size(); ++i) {
    if (prefill_input_ptrs[i] != nullptr) {
      delete prefill_input_ptrs[i];
      prefill_input_ptrs[i] = nullptr;
    }
  }

  for (int i = 0; i < decode_input_ptrs.size(); ++i) {
    if (decode_input_ptrs[i] != nullptr) {
      delete decode_input_ptrs[i];
      decode_input_ptrs[i] = nullptr;
    }
  }

  for (int i = 0; i < vit_input_ptrs.size(); ++i) {
    if (vit_input_ptrs[i] != nullptr) {
      delete vit_input_ptrs[i];
      vit_input_ptrs[i] = nullptr;
    }
  }

  prefill_module.reset();
  decode_module.reset();
  vit_module.reset();
}

void HmvllmInfer::PrefillSetInputDatas(void *data) {
  prefill_input_map.clear();
  for (int idx = 0; idx < attn_idx_start; idx++) {
    auto input_name = prefill_module->GetInputName(idx);
    auto input_info = prefill_module->GetInputInfo(input_name).AsContiguous();

    tcim::Tensor input_tensor;
    size_t mem_size = 0;
    if (idx == 0) {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
    } else {
      mem_size = input_info.MemSize();
      if (prefill_input_ptrs[idx - 1] == nullptr) {
        prefill_input_ptrs[idx - 1] = new char[mem_size];
        memset(prefill_input_ptrs[idx - 1], 0, mem_size);
      }
      input_tensor = tcim::Tensor::CreateHostTensor(
          input_info, mem_size, prefill_input_ptrs[idx - 1]);
    }

    if (prefill_input_map.find(input_name) != prefill_input_map.end()) {
      prefill_input_map.at(input_name) = input_tensor;
    } else {
      prefill_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
  }

  for (const auto &input : prefill_input_map) {
    prefill_module->SetInput(input.first, input.second);
  }
}

float HmvllmInfer::PrefillInfer() {
  auto t_start = std::chrono::high_resolution_clock::now();
  prefill_module->Run();
  prefill_module->Sync();
  auto t_end = std::chrono::high_resolution_clock::now();
  float t_total =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
  return t_total;
}

void HmvllmInfer::PrefillGetOutputDatas(std::vector<int32_t> &ids) {
  int output_num = prefill_module->GetOutputNum();

  auto output_name = prefill_module->GetOutputName(0);
  auto output_info = prefill_module->GetOutputInfo(output_name).AsContiguous();

  auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
  prefill_output_map.insert(
      std::pair<std::string, tcim::Tensor>(output_name, output_tensor));

  auto output = *prefill_output_map.begin();
  output_tensor = prefill_module->GetOutput(output.first);
  output_tensor.CastTo(output.second);

  void *prefill_outData = output_tensor.Data();
  ids.emplace_back(eigen_argmax<tensor_type>(
      static_cast<tensor_type *>(prefill_outData), argmax_dim_len));
}

void HmvllmInfer::DecodeSetInputDatas(void *data, int valid_length) {
  for (int idx = 0; idx < attn_idx_start; idx++) {
    auto input_name = decode_module->GetInputName(idx);
    auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();

    tcim::Tensor input_tensor;
    size_t mem_size = 0;
    if (idx == 0) {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
    } else {
      mem_size = input_info.MemSize();
      if (decode_input_ptrs[idx - 1] == nullptr) {
        decode_input_ptrs[idx - 1] = new char[mem_size];
        memset(decode_input_ptrs[idx - 1], 0, mem_size);
        if (input_name == "valid_length") {
          memcpy(decode_input_ptrs[idx - 1], &valid_length,
                 sizeof(valid_length));
        }
      }
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                    decode_input_ptrs[idx - 1]);
    }

    if (decode_input_map.find(input_name) != decode_input_map.end()) {
      decode_input_map.at(input_name) = input_tensor;
    } else {
      decode_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
  }

  for (const auto &input : decode_input_map) {
    decode_module->SetInput(input.first, input.second);
  }
}

float HmvllmInfer::DecodeInfer() {
  auto t_start = std::chrono::high_resolution_clock::now();
  decode_module->Run();
  decode_module->Sync();
  auto t_end = std::chrono::high_resolution_clock::now();
  float t_total =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
  return t_total;
}

void HmvllmInfer::DecodeGetOutputDatas(std::vector<int32_t> &ids) {
  int output_num = decode_module->GetOutputNum();
  auto output_name = decode_module->GetOutputName(0);
  auto output_info = decode_module->GetOutputInfo(output_name).AsContiguous();

  auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);

  if (decode_output_map.find(output_name) != decode_output_map.end()) {
    decode_output_map.at(output_name) = output_tensor;
  } else {
    decode_output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }

  auto output = *decode_output_map.begin();
  output_tensor = decode_module->GetOutput(output.first);
  output_tensor.CastTo(output.second);

  void *decode_outData = output_tensor.Data();
  ids.emplace_back(eigen_argmax<tensor_type>(
      static_cast<tensor_type *>(decode_outData), argmax_dim_len));
}

void HmvllmInfer::VitSetInput() {
  for (int idx = 0; idx < vit_input_nums; idx++) {
    auto input_name = vit_module->GetInputName(idx);
    auto input_info = vit_module->GetInputInfo(input_name).AsContiguous();

    tcim::Tensor input_tensor;
    size_t mem_size = 0;

    mem_size = input_info.MemSize();
    if (vit_input_ptrs[idx] == nullptr) {
      vit_input_ptrs[idx] = new char[mem_size];
      memset(vit_input_ptrs[idx], 0, mem_size);
    }
    input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                  vit_input_ptrs[idx]);

    if (vit_input_map.find(input_name) != vit_input_map.end()) {
      vit_input_map.at(input_name) = input_tensor;
    } else {
      vit_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
  }

  for (const auto &input : vit_input_map) {
    vit_module->SetInput(input.first, input.second);
  }
}

float HmvllmInfer::VitInfer() {
  auto t_start = std::chrono::high_resolution_clock::now();
  vit_module->Run();
  vit_module->Sync();
  auto t_end = std::chrono::high_resolution_clock::now();
  float t_total =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
  return t_total;
}

void HmvllmInfer::VitGetOutputDatas() {
  int output_num = vit_module->GetOutputNum();
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = vit_module->GetOutputName(idx);
    auto output_info = vit_module->GetOutputInfo(output_name).AsContiguous();
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    if (vit_output_map.find(output_name) != vit_output_map.end()) {
      vit_output_map.at(output_name) = output_tensor;
    } else {
      vit_output_map.insert(
          std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }

    output_tensor = vit_module->GetOutput(output_name);
    output_tensor.CastTo(vit_output_map.at(output_name));
  }
}

PerfInfos HmvllmInfer::perf_llm(const uint32_t input_tokens_len,
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
  std::vector<int> all_input_ids = generateRandomVector(input_tokens_len);
  PerfInfos vllm_perf_datas;
  memset(&vllm_perf_datas, 0, sizeof(PerfInfos));
  VitSetInput();
  vllm_perf_datas.vit_time = VitInfer();
  VitGetOutputDatas();
  auto t_start = std::chrono::high_resolution_clock::now();
  auto t_embed_start = std::chrono::high_resolution_clock::now();
  auto t_embed_end = std::chrono::high_resolution_clock::now();
  auto t_ttft_end = std::chrono::high_resolution_clock::now();
  auto t_decode_end = std::chrono::high_resolution_clock::now();
  auto t_ttft_start = std::chrono::high_resolution_clock::now();
  vllm_perf_datas.input_tokens = input_tokens_len;
  if (input_tokens_len + stop_tokens_len > context_max_length) {
    std::cout << "input_tokens_len + stop_tokens_len > context_max_length, "
                 "cast stop_tokens_len to "
              << context_max_length - input_tokens_len << std::endl;
    vllm_perf_datas.stop_tokens = context_max_length - input_tokens_len;
  } else {
    vllm_perf_datas.stop_tokens = stop_tokens_len;
  }
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

    t_embed_start = std::chrono::high_resolution_clock::now();
    input_datas = embedding->EmbeddingTokens(input_ids);
    t_embed_end = std::chrono::high_resolution_clock::now();
    vllm_perf_datas.embedding_time +=
        std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
            .count();

    PrefillSetInputDatas(input_datas);
    vllm_perf_datas.prefill_time += PrefillInfer();
  }

  PrefillGetOutputDatas(ids);
  t_ttft_end = std::chrono::high_resolution_clock::now();
  vllm_perf_datas.ttft +=
      std::chrono::duration<float, std::milli>(t_ttft_end - t_ttft_start)
          .count();
  int context_length = input_tokens_len;

  do {
    if ((context_length > context_max_length) ||
        (vllm_perf_datas.decode_count >= vllm_perf_datas.stop_tokens)) {
      break;
    }

    t_embed_start = std::chrono::high_resolution_clock::now();
    input_datas = embedding->EmbeddingTokens(ids);
    t_embed_end = std::chrono::high_resolution_clock::now();
    vllm_perf_datas.embedding_time +=
        std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
            .count();

    DecodeSetInputDatas(static_cast<void *>(input_datas), context_length);
    vllm_perf_datas.decode_time += DecodeInfer();
    ids.clear();
    DecodeGetOutputDatas(ids);
    vllm_perf_datas.decode_count++;

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

  auto t_end = std::chrono::high_resolution_clock::now();
  vllm_perf_datas.t_total =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
  // perf information
  ShowPerfInformation(vllm_perf_datas);
  return vllm_perf_datas;
}