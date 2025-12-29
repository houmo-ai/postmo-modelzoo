#include "HmllmInferMultiBatch.h"

HmllmInferMultiBatch::HmllmInferMultiBatch(
    const std::string &prefillModelPath, const std::string &decodeModelPath,
    const std::string &embeddingWeightPath, int ndevices, int batches) {
  this->prefillModelPath = prefillModelPath;
  this->decodeModelPath = decodeModelPath;
  // 创建weightManager
  std::vector<int> devs;
  devs.clear();
  std::cout << "Multi batch Use Devices ";
  for (int i = 0; i < ndevices; i++) {
    devs.emplace_back(i);
    std::cout << i << " ";
  }
  std::cout << std::endl;
  tcim::DevManager dev_manager = tcim::DevManager::Create(devs);
  weight_manager =
      tcim::Module::WeightManager::CreateWeightManager(dev_manager);
  // 创建weightManager
  auto option_prefill = tcim::Module::Option(weight_manager);
  auto option_decode = tcim::Module::Option(weight_manager);
  option_prefill.EnableLazyMode(true);
  option_decode.EnableLazyMode(true);
  // 初始化Module
  prefill_module = std::make_shared<tcim::Module>();
  prefill_module->LoadModel(prefillModelPath, option_prefill);
  decode_module = std::make_shared<tcim::Module>();
  decode_module->LoadModel(decodeModelPath, option_decode);

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

  option_prefill.SetDummyTensors(dummy_names);

  // 获取模型配置
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
  this->argmax_dim_len =
      decode_module->GetOutputInfo(decode_module->GetOutputName(0)).Shape()[2];
  this->next_ids.clear();
  this->current_echo_lens.clear();

  for (int idx = 0; idx < this->batch; idx++) {
    this->next_ids.emplace_back(0);
    this->current_echo_lens.emplace_back(0);
  }

  // 配置Decode其他输入
  for (int b = 0; b < this->batch; ++b) {
    int index = (b == 0) ? 2 : (2 * n_blocks * b + 3 + 2 * b - 1);
    auto input_name = decode_module->GetInputName(index);
    auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();
    size_t mem_size = input_info.MemSize();
    tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(
        input_info, mem_size, &decode_current_length);
    decode_module->SetInput(input_name, input_tensor);
  }

  // embedding
  embedding = std::make_shared<HmEmbedding>(
      embeddingWeightPath, this->embedding_length, this->prefill_length);
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
#ifdef BACKEND_XH1
  int16_t valid_length_int16t = valid_length;
  int16_t current_length_int16t = current_length;
#endif
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
#ifdef BACKEND_XH1
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                    &valid_length_int16t);
#else
      input_tensor =
          tcim::Tensor::CreateHostTensor(input_info, mem_size, &valid_length);
#endif
    } else if (idx == 2) {
      mem_size = input_info.MemSize();
#ifdef BACKEND_XH1
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                    &current_length_int16t);
#else
      input_tensor =
          tcim::Tensor::CreateHostTensor(input_info, mem_size, &current_length);
#endif
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
#ifdef BACKEND_XH1
  int16_t context_length_int16t = context_length;
#endif
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
#ifdef BACKEND_XH1
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                    &context_length_int16t);
#else
      input_tensor =
          tcim::Tensor::CreateHostTensor(input_info, mem_size, &context_length);
#endif
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
  ret.time = 0.f;
  ret.embedding_time = 0.f;
  ret.next_id.clear();

  auto t_embed_start = std::chrono::high_resolution_clock::now();
  auto t_embed_end = std::chrono::high_resolution_clock::now();

  int decode_input_index_start =
      (batch > 0) ? (2 * this->n_blocks * batch + 3 + 2 * batch) : 3;
  int decode_input_index_finish =
      2 * this->n_blocks * (batch + 1) + 3 + batch * 2;
  int prefill_input_index = 3;

  for (int i = decode_input_index_start; i < decode_input_index_finish; ++i) {
    tcim::Tensor cache =
        decode_module->GetDevInput(decode_module->GetInputName(i));
    prefill_module->SetInput(prefill_module->GetInputName(prefill_input_index),
                             cache);
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
    t_embed_start = std::chrono::high_resolution_clock::now();
    input_datas = embedding->EmbeddingTokens(input_ids);
    t_embed_end = std::chrono::high_resolution_clock::now();
    ret.embedding_time +=
        std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
            .count();
    PrefillSetInputDatas(input_datas, valid_length, current_length);
    auto t_prefill_end = std::chrono::high_resolution_clock::now();
    auto t_prefill_start = std::chrono::high_resolution_clock::now();
    PrefillInfer();
    t_prefill_end = std::chrono::high_resolution_clock::now();
    ret.time += std::chrono::duration<float, std::milli>(t_prefill_end -
                                                         t_prefill_start)
                    .count();
  }

  PrefillGetOutputDatas(ret.next_id);

  return ret;
}

PerfSingleBatchInfo HmllmInferMultiBatch::run_decode(
    tensor_type *input_datas, const std::vector<int> context_length) {
  PerfSingleBatchInfo ret;
  ret.time = 0.f;
  ret.embedding_time = 0.f;

  auto input_name = decode_module->GetInputName(0);
  auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();
  size_t mem_size = input_info.MemSize();
  tcim::Tensor input_tensor = tcim::Tensor::CreateHostTensor(
      input_info, mem_size, static_cast<void *>(input_datas));

  decode_module->SetInput(input_name, input_tensor);

  for (int b = 0; b < this->batch; ++b) {
    int valid_length_index = (b == 0) ? 1 : (2 * n_blocks * b + 3 + 2 * b - 2);
    int32_t valid_length_data = context_length[b];
    input_name = decode_module->GetInputName(valid_length_index);
    input_info = decode_module->GetInputInfo(input_name).AsContiguous();
    mem_size = input_info.MemSize();
    input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                  &valid_length_data);
    decode_module->SetInput(input_name, input_tensor);
  }
  auto t_decode_end = std::chrono::high_resolution_clock::now();
  auto t_decode_start = std::chrono::high_resolution_clock::now();
  decode_module->Run();
  decode_module->Sync();
  t_decode_end = std::chrono::high_resolution_clock::now();
  ret.time +=
      std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start)
          .count();
  DecodeGetOutputDatas();
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
  auto t_embed_start = std::chrono::high_resolution_clock::now();
  auto t_embed_end = std::chrono::high_resolution_clock::now();
  auto t_total_start = std::chrono::high_resolution_clock::now();
  auto t_ttft_start = std::chrono::high_resolution_clock::now();
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
    PerfSingleBatchInfo retInfo = run_prefill(b, all_input_ids);
    llm_perf_datas.prefill_time += retInfo.time;
    next_ids[b] = retInfo.next_id;
    current_echo_lens[b] = retInfo.input_tokens;
    llm_perf_datas.input_tokens += retInfo.input_tokens;
    t_embed_start = std::chrono::high_resolution_clock::now();
    tensor_type *input_data = embedding->EmbeddingTokens(next_ids[b]);
    memcpy(static_cast<void *>(&input_datas[b * embedding_length]),
           static_cast<void *>(input_data),
           embedding_length * sizeof(tensor_type));
    t_embed_end = std::chrono::high_resolution_clock::now();
    llm_perf_datas.embedding_time +=
        std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
            .count();
  }
  auto t_ttft_end = std::chrono::high_resolution_clock::now();
  llm_perf_datas.ttft +=
      std::chrono::duration<float, std::milli>(t_ttft_end - t_ttft_start)
          .count();
  std::vector<int> context_length = current_echo_lens;
  do {
    if ((llm_perf_datas.decode_count >= llm_perf_datas.stop_tokens)) {
      break;
    }

    PerfSingleBatchInfo retInfo = run_decode(input_datas, context_length);
    llm_perf_datas.decode_time += retInfo.time;
    for (int b = 0; b < this->batch; ++b) {
      context_length[b] += 1;
      t_embed_start = std::chrono::high_resolution_clock::now();
      tensor_type *input_data = embedding->EmbeddingTokens(next_ids[b]);
      memcpy(static_cast<void *>(&input_datas[b * embedding_length]),
             static_cast<void *>(input_data),
             embedding_length * sizeof(tensor_type));
      t_embed_end = std::chrono::high_resolution_clock::now();
      llm_perf_datas.embedding_time +=
          std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
              .count();
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
  // perf information
  auto t_total_end = std::chrono::high_resolution_clock::now();
  llm_perf_datas.t_total +=
      std::chrono::duration<float, std::milli>(t_total_end - t_total_start)
          .count();
  llm_perf_datas.decode_count = llm_perf_datas.decode_count * this->batch;
  ShowPerfInformation(llm_perf_datas);
  return llm_perf_datas;
}