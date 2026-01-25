/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmllmInferMultiBatch.cc
 * Description:
 *   HmllmInferMultiBatch Implementation - Multi-batch performance testing
 * implementation for large language model inference.
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
#include "HmllmInferMultiBatch.h"

HmllmInferMultiBatch::HmllmInferMultiBatch(
    const std::string &prefillModelPath, const std::string &decodeModelPath,
    const std::string &embeddingWeightPath, int ndevices, int batches,
    bool LazyMode) {
  perf_tracker = std::make_shared<InferencePerformanceTracker>();
  this->prefillModelPath = prefillModelPath;
  this->decodeModelPath = decodeModelPath;

  // Create weightManager
  std::vector<int> devs;
  devs.clear();
  std::cout << "Multi batch Use Devices ";
  for (int i = 0; i < ndevices; i++) {
    devs.emplace_back(i);
    std::cout << i << " ";
  }
  std::cout << std::endl;

  // Create device manager and weight manager
  tcim::DevManager dev_manager = tcim::DevManager::Create(devs);
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
  // Initialize Module - Load prefill and decode models
  prefill_module = std::make_shared<tcim::Module>();
  perf_tracker->perfStart(PerfType::PREFILL_LOAD_TIME);
  prefill_module->LoadModel(prefillModelPath, option_prefill);
  perf_tracker->perfEnd(PerfType::PREFILL_LOAD_TIME);
  decode_module = std::make_shared<tcim::Module>();
  perf_tracker->perfStart(PerfType::DECODE_LOAD_TIME);
  decode_module->LoadModel(decodeModelPath, option_decode);
  perf_tracker->perfEnd(PerfType::DECODE_LOAD_TIME);

  // Get number of blocks in the model and create dummy names for cache inputs
  n_blocks = get_nblocks();
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

  // Set dummy tensors for prefill module
  option_prefill.SetDummyTensors(dummy_names);

  // Get model configuration parameters
  this->prefill_length =
      prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[1];
  this->embedding_length =
      prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[2];
  this->context_max_length =
      prefill_module->GetInputInfo(prefill_module->GetInputName(3)).Shape()[2];
  this->batch =
      decode_module->GetInputInfo(decode_module->GetInputName(0)).Shape()[0];

  if (this->batch != batches) {
    throw std::runtime_error("Model Batch Not match args batch!");
  }

  // Get the dimension length for argmax operation from decode module output
  this->argmax_dim_len =
      decode_module->GetOutputInfo(decode_module->GetOutputName(0)).Shape()[2];

  // Clear and initialize the next_ids and current_echo_lens vectors
  this->next_ids.clear();
  this->current_echo_lens.clear();

  // Initialize vectors with zeros based on batch size
  for (int idx = 0; idx < this->batch; idx++) {
    this->next_ids.emplace_back(0);
    this->current_echo_lens.emplace_back(0);
  }

  // Configure additional inputs for decode module (KV cache inputs)
  for (int b = 0; b < this->batch; ++b) {
    // Calculate the index for the decode current length input
    int index = (b == 0) ? 2 : (2 * n_blocks * b + 3 + 2 * b - 1);
    auto input_name = decode_module->GetInputName(index);
    auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();
    size_t mem_size = input_info.MemSize();
    // Create tensor for current length input and set it to the decode module
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(
        input_info, mem_size, &decode_current_length);
    decode_module->SetInput(input_name, input_tensor);
  }

  // Initialize embedding module with the specified parameters
  embedding = std::make_shared<HmEmbedding>(
      embeddingWeightPath, this->embedding_length, this->prefill_length);
  perf_tracker->reset();
}

int HmllmInferMultiBatch::get_nblocks() {
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

void HmllmInferMultiBatch::DebugModelInfo(tcim::Module &module,
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

HmllmInferMultiBatch::~HmllmInferMultiBatch() {
  prefill_module.reset();
  decode_module.reset();
  embedding.reset();
}

void HmllmInferMultiBatch::PrefillSetInputDatas(void *data,
                                                int32_t valid_length,
                                                int32_t current_length) {
  prefill_input_map.clear();
  for (int idx = 0; idx < 3; idx++) {
    auto input_name = prefill_module->GetInputName(idx);
    auto input_info = prefill_module->GetInputInfo(input_name).AsContiguous();

    tcim::Tensor input_tensor;
    size_t mem_size = 0;
    if (idx == 0) {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
    } else if (idx == 1) {
      mem_size = input_info.MemSize();
      input_tensor =
          tcim::Tensor::CreateHostTensor(input_info, mem_size, &valid_length);
    } else if (idx == 2) {
      mem_size = input_info.MemSize();
      input_tensor =
          tcim::Tensor::CreateHostTensor(input_info, mem_size, &current_length);
    } else {
      break;
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

void HmllmInferMultiBatch::PrefillInfer() {
  prefill_module->Run();
  prefill_module->Sync();
}

void HmllmInferMultiBatch::PrefillGetOutputDatas(std::vector<int32_t> &ids) {
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

void HmllmInferMultiBatch::DecodeSetInputDatas(void *data,
                                               int32_t context_length) {
  for (int idx = 0; idx < 2; idx++) {
    auto input_name = decode_module->GetInputName(idx);
    auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();

    tcim::Tensor input_tensor;
    size_t mem_size = 0;
    if (idx == 0) {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
    } else if (idx == 1) {
      mem_size = input_info.MemSize();
      input_tensor =
          tcim::Tensor::CreateHostTensor(input_info, mem_size, &context_length);
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

void HmllmInferMultiBatch::DecodeInfer() {
  decode_module->Run();
  decode_module->Sync();
}

void HmllmInferMultiBatch::DecodeGetOutputDatas() {
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
  next_ids.clear();
  next_ids.resize(this->batch);
  for (int b = 0; b < this->batch; ++b) {
    std::vector<int> ids;
    ids.emplace_back(eigen_argmax<tensor_type>(
        reinterpret_cast<tensor_type *>(
            reinterpret_cast<char *>(decode_outData) +
            b * this->embedding_length * sizeof(tensor_type)),
        argmax_dim_len));
    next_ids[b] = ids;
  }
}

PerfSingleBatchInfo HmllmInferMultiBatch::run_prefill(
    int batch, const std::vector<int> all_input_ids) {
  PerfSingleBatchInfo ret;
  ret.input_tokens = all_input_ids.size();
  int decode_input_index_start =
      (batch > 0) ? (2 * this->n_blocks * batch + 3 + 2 * batch) : 3;
  int decode_input_index_finish =
      2 * this->n_blocks * (batch + 1) + 3 + batch * 2;
  int prefill_input_index = 3;

  for (int i = decode_input_index_start; i < decode_input_index_finish; ++i) {
    tcim::Tensor cache =
        decode_module->GetDevInput(decode_module->GetInputName(i));
    perf_tracker->perfStart(PerfType::PREFILL_INPUT_TIME);
    prefill_module->SetInput(prefill_module->GetInputName(prefill_input_index),
                             cache);
    perf_tracker->perfEnd(PerfType::PREFILL_INPUT_TIME);
    prefill_input_index++;
  }

  int input_tokens_len = ret.input_tokens;
  if (input_tokens_len > this->context_max_length) {
    throw std::runtime_error("Question long than " +
                             std::to_string(context_max_length) +
                             ", please shorten it !");
  }
  int prefill_loop_round =
      std::ceil((float)input_tokens_len / (float)prefill_length);

  int32_t valid_length = 0, current_length = 0;
  tensor_type *input_datas = nullptr;

  for (int round = 0; round < prefill_loop_round; ++round) {
    valid_length = round * this->prefill_length;
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

    int32_t effective_length = input_ids.size();
    perf_tracker->perfStart(PerfType::PREFILL_EMBED_TIME);
    input_datas = embedding->EmbeddingTokens(input_ids);
    perf_tracker->perfEnd(PerfType::PREFILL_EMBED_TIME);

    perf_tracker->perfStart(PerfType::PREFILL_INPUT_TIME);
    PrefillSetInputDatas(input_datas, valid_length, current_length);
    perf_tracker->perfEnd(PerfType::PREFILL_INPUT_TIME);

    perf_tracker->perfStart(PerfType::PREFILL_INFER_TIME);
    PrefillInfer();
    perf_tracker->perfEnd(PerfType::PREFILL_INFER_TIME);
  }

  perf_tracker->perfStart(PerfType::PREFILL_OUTPUT_TIME);
  PrefillGetOutputDatas(ret.next_id);
  perf_tracker->perfEnd(PerfType::PREFILL_OUTPUT_TIME);
  return ret;
}

PerfSingleBatchInfo HmllmInferMultiBatch::run_decode(
    tensor_type *input_datas, const std::vector<int> context_length) {
  PerfSingleBatchInfo ret;

  auto input_name = decode_module->GetInputName(0);
  auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();
  size_t mem_size = input_info.MemSize();
  tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(
      input_info, mem_size, static_cast<void *>(input_datas));
  perf_tracker->perfStart(PerfType::DECODE_INPUT_TIME);
  decode_module->SetInput(input_name, input_tensor);
  perf_tracker->perfEnd(PerfType::DECODE_INPUT_TIME);

  for (int b = 0; b < this->batch; ++b) {
    int valid_length_index = (b == 0) ? 1 : (2 * n_blocks * b + 3 + 2 * b - 2);
    int32_t valid_length_data = context_length[b];
    input_name = decode_module->GetInputName(valid_length_index);
    input_info = decode_module->GetInputInfo(input_name).AsContiguous();
    mem_size = input_info.MemSize();
    input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                  &valid_length_data);
    perf_tracker->perfStart(PerfType::DECODE_INPUT_TIME);
    decode_module->SetInput(input_name, input_tensor);
    perf_tracker->perfEnd(PerfType::DECODE_INPUT_TIME);
  }

  perf_tracker->perfStart(PerfType::DECODE_INFER_TIME);
  decode_module->Run();
  decode_module->Sync();
  perf_tracker->perfEnd(PerfType::DECODE_INFER_TIME);

  perf_tracker->perfStart(PerfType::DECODE_OUTPUT_TIME);
  DecodeGetOutputDatas();
  perf_tracker->perfEnd(PerfType::DECODE_OUTPUT_TIME);
  return ret;
}

PerfInfos HmllmInferMultiBatch::perf_llm(const uint32_t input_tokens_len,
                                         const uint32_t stop_tokens_len) {
  if (input_tokens_len > context_max_length) {
    throw std::runtime_error("Question long than " +
                             std::to_string(context_max_length) +
                             ", please shorten it !");
  }
  tensor_type *input_datas =
      new tensor_type[this->batch * this->embedding_length];
  memset(input_datas, 0, this->batch * this->embedding_length);
  PerfInfos llm_perf_datas;
  memset(&llm_perf_datas, 0, sizeof(PerfInfos));

  if (input_tokens_len + stop_tokens_len > context_max_length) {
    std::cout << "input_tokens_len + stop_tokens_len > context_max_length, "
                 "cast stop_tokens_len to "
              << context_max_length - input_tokens_len << std::endl;
    llm_perf_datas.stop_tokens = context_max_length - input_tokens_len;
  } else {
    llm_perf_datas.stop_tokens = stop_tokens_len;
  }
  next_ids.resize(this->batch);
  std::vector<int> current_echo_lens;
  current_echo_lens.resize(this->batch);
  for (int b = 0; b < this->batch; ++b) {
    std::vector<int> all_input_ids = generateRandomVector(input_tokens_len);
    perf_tracker->perfStart(PerfType::PREFILL_TOTAL_TIME);
    PerfSingleBatchInfo retInfo = run_prefill(b, all_input_ids);
    perf_tracker->perfEnd(PerfType::PREFILL_TOTAL_TIME);
    next_ids[b] = retInfo.next_id;
    current_echo_lens[b] = retInfo.input_tokens;
    llm_perf_datas.input_tokens += retInfo.input_tokens;

    perf_tracker->perfStart(PerfType::DECODE_EMBED_TIME);
    tensor_type *input_data = embedding->EmbeddingTokens(next_ids[b]);
    memcpy(static_cast<void *>(&input_datas[b * embedding_length]),
           static_cast<void *>(input_data),
           embedding_length * sizeof(tensor_type));

    perf_tracker->perfEnd(PerfType::DECODE_EMBED_TIME);
  }

  std::vector<int> context_length = current_echo_lens;
  do {
    if ((llm_perf_datas.decode_count >= llm_perf_datas.stop_tokens)) {
      break;
    }

    perf_tracker->perfStart(PerfType::DECODE_TOTAL_TIME);
    run_decode(input_datas, context_length);
    perf_tracker->perfEnd(PerfType::DECODE_TOTAL_TIME);

    for (int b = 0; b < this->batch; ++b) {
      context_length[b] += 1;
      perf_tracker->perfStart(PerfType::DECODE_EMBED_TIME);
      tensor_type *input_data = embedding->EmbeddingTokens(next_ids[b]);
      memcpy(static_cast<void *>(&input_datas[b * embedding_length]),
             static_cast<void *>(input_data),
             embedding_length * sizeof(tensor_type));
      perf_tracker->perfEnd(PerfType::DECODE_EMBED_TIME);
    }

    llm_perf_datas.decode_count++;

    double ratio = static_cast<double>(llm_perf_datas.decode_count) /
                   llm_perf_datas.stop_tokens;
    int filled = static_cast<int>(ratio * bar_width);
    std::cout << '\r' << "Decode: " << std::setw(3) << int(ratio * 100) << "% |"
              << std::string(filled, '*')
              << std::string(bar_width - filled, ' ') << "| "
              << llm_perf_datas.decode_count << '/'
              << llm_perf_datas.stop_tokens << std::flush;
  } while (true);

  perf_tracker->setBasicInfo(this->batch, input_tokens_len,
                             llm_perf_datas.decode_count);
  perf_tracker->showSummary();
  return llm_perf_datas;
}