#include "HmllmInfer.h"

HmllmInfer::HmllmInfer(const std::string &prefillModelPath,
                       const std::string &decodeModelPath,
                       const std::string &embeddingWeightPath, int ndevices,
                       int batches) {
  this->prefillModelPath = prefillModelPath;
  this->decodeModelPath = decodeModelPath;
  // 创建weightManager
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
  // 创建weightManager
  auto option_prefill = tcim::Module::Option(weight_manager);
  auto option_decode = tcim::Module::Option(weight_manager);
  option_prefill.EnableLazyMode(true);
  option_decode.EnableLazyMode(true);
  // 初始化Module
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

  option_decode.SetDummyTensors(dummy_names);
  decode_module = std::make_shared<tcim::Module>();
  decode_module->LoadModel(decodeModelPath, option_decode);

  // 获取模型配置
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
  // 配置Decode其他输入
  for (int idx = attn_idx_start; idx < 2 * n_blocks + attn_idx_start; idx++) {
    const std::string input_name = prefill_module->GetInputName(idx);
    auto cache = prefill_module->GetInput(input_name);
    decode_input_map.insert(std::pair<std::string, tcim::Tensor>(
        decode_module->GetInputName(idx), cache));
  }
  // set decode current_length input
  const std::string decode_current_length_name = decode_module->GetInputName(2);
  const tcim::TensorInfo decode_current_length_input_info =
      decode_module->GetInputInfo(decode_current_length_name).AsContiguous();
  size_t decode_current_length_mem_size =
      decode_current_length_input_info.MemSize();
  tcim::Tensor decode_current_length_input_tensor =
      tcim::Tensor::CreateHostTensor(decode_current_length_input_info,
                                     decode_current_length_mem_size,
                                     &decode_current_length);
  decode_input_map.insert(std::pair<std::string, tcim::Tensor>(
      decode_current_length_name, decode_current_length_input_tensor));

  // embedding
  embedding = std::make_shared<HmEmbedding>(
      embeddingWeightPath, this->embedding_length, this->prefill_length);
  // DebugModelInfo(*prefill_module.get(), prefillModelPath);
  // DebugModelInfo(*decode_module.get(), decodeModelPath);
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
  prefill_input_map.clear();

  for (int idx = 0; idx < attn_idx_start; idx++) {
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
    } else if (idx == 3) {
      // only support Qwen3-30b
      mem_size = input_info.MemSize();
      std::vector<int32_t> position_ids;
      for (int i = valid_length; i < valid_length + this->prefill_length; ++i) {
        position_ids.emplace_back(i);
      }
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size,
                                                    position_ids.data());
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

float HmllmInfer::PrefillInfer() {
  auto t_start = std::chrono::high_resolution_clock::now();
  prefill_module->Run();
  prefill_module->Sync();
  auto t_end = std::chrono::high_resolution_clock::now();
  float t_total =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
  return t_total;
}

void HmllmInfer::PrefillGetOutputDatas(std::vector<int32_t> &ids) {
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

void HmllmInfer::DecodeSetInputDatas(void *data, int32_t context_length) {
  for (int idx = 0; idx < attn_idx_start; idx++) {
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
    } else if (idx == 2) {
      continue;
    } else if (idx == 3) {
      mem_size = input_info.MemSize();
      int32_t position_id = context_length + 1;
      input_tensor =
          tcim::Tensor::CreateHostTensor(input_info, mem_size, &position_id);
    } else {
      break;
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

float HmllmInfer::DecodeInfer() {
  auto t_start = std::chrono::high_resolution_clock::now();
  decode_module->Run();
  decode_module->Sync();
  auto t_end = std::chrono::high_resolution_clock::now();
  float t_total =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
  return t_total;
}

void HmllmInfer::DecodeGetOutputDatas(std::vector<int32_t> &ids) {
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

PerfInfos HmllmInfer::perf_llm(const uint32_t input_tokens_len,
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
  PerfInfos llm_perf_datas;
  memset(&llm_perf_datas, 0, sizeof(PerfInfos));
  auto t_start = std::chrono::high_resolution_clock::now();
  auto t_embed_start = std::chrono::high_resolution_clock::now();
  auto t_embed_end = std::chrono::high_resolution_clock::now();
  auto t_ttft_end = std::chrono::high_resolution_clock::now();
  auto t_decode_start = std::chrono::high_resolution_clock::now();
  auto t_decode_end = std::chrono::high_resolution_clock::now();

  auto t_ttft_start = std::chrono::high_resolution_clock::now();
  llm_perf_datas.input_tokens = input_tokens_len;
  if (input_tokens_len + stop_tokens_len > context_max_length) {
    std::cout << "input_tokens_len + stop_tokens_len > context_max_length, "
                 "cast stop_tokens_len to "
              << context_max_length - input_tokens_len << std::endl;
    llm_perf_datas.stop_tokens = context_max_length - input_tokens_len;
  } else {
    llm_perf_datas.stop_tokens = stop_tokens_len;
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
    llm_perf_datas.embedding_time +=
        std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
            .count();

    PrefillSetInputDatas(input_datas, valid_length, current_length);
    llm_perf_datas.prefill_time += PrefillInfer();
  }

  PrefillGetOutputDatas(ids);
  t_ttft_end = std::chrono::high_resolution_clock::now();
  llm_perf_datas.ttft +=
      std::chrono::duration<float, std::milli>(t_ttft_end - t_ttft_start)
          .count();
  int context_length = input_tokens_len;

  do {
    if ((context_length > context_max_length) ||
        (llm_perf_datas.decode_count >= llm_perf_datas.stop_tokens)) {
      break;
    }

    t_embed_start = std::chrono::high_resolution_clock::now();
    input_datas = embedding->EmbeddingTokens(ids);
    t_embed_end = std::chrono::high_resolution_clock::now();
    llm_perf_datas.embedding_time +=
        std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
            .count();

    DecodeSetInputDatas(static_cast<void *>(input_datas),
                        static_cast<int32_t>(context_length));
    llm_perf_datas.decode_time += DecodeInfer();
    ids.clear();
    DecodeGetOutputDatas(ids);
    llm_perf_datas.decode_count++;

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
  auto t_end = std::chrono::high_resolution_clock::now();
  llm_perf_datas.t_total =
      std::chrono::duration<float, std::milli>(t_end - t_start).count();
  // perf information
  ShowPerfInformation(llm_perf_datas);
  return llm_perf_datas;
}