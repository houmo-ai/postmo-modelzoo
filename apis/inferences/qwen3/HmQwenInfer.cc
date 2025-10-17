#include "HmQwenInfer.h"

HmQwenInfer::HmQwenInfer(const std::string &prefillModelPath,
                         const std::string &decodeModelPath,
                         const std::string &tokenizerJsonPath,
                         const std::string &embeddingWeightPath)
{
  this->prefillModelPath = prefillModelPath;
  this->decodeModelPath = decodeModelPath;
  // 创建weightManager
  weight_manager = tcim::Module::WeightManager::CreateWeightManager(0);
  auto option_prefill = tcim::Module::Option(weight_manager);
  auto option_decode = tcim::Module::Option(weight_manager);
  // 初始化Module
  prefill_module = std::make_shared<tcim::Module>();
  prefill_module->LoadModel(prefillModelPath, option_prefill);
  decode_module = std::make_shared<tcim::Module>();
  decode_module->LoadModel(decodeModelPath, option_decode);
  // 获取模型配置
  this->prefill_length = prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[1];
  this->embedding_length = prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[2];
  this->context_max_length = prefill_module->GetInputInfo(prefill_module->GetInputName(3)).Shape()[2];
  this->batch = decode_module->GetInputInfo(decode_module->GetInputName(0)).Shape()[0];
  this->argmax_dim_len = decode_module->GetOutputInfo(decode_module->GetOutputName(0)).Shape()[2];
  // 配置Decode其他输入
  int n_blocks = get_nblocks();
  for (int idx = 3; idx < 2 * n_blocks + 3; idx++)
  {
    const std::string input_name = prefill_module->GetInputName(idx);
    auto cache = prefill_module->GetInput(input_name);
    decode_input_map.insert(
        std::pair<std::string, tcim::Tensor>(decode_module->GetInputName(idx), cache));
  }
  // set decode current_length input
  const std::string decode_current_length_name = decode_module->GetInputName(2);
  const tcim::TensorInfo decode_current_length_input_info = decode_module->GetInputInfo(decode_current_length_name).AsContiguous();
  size_t decode_current_length_mem_size = decode_current_length_input_info.MemSize();
  tcim::Tensor decode_current_length_input_tensor = tcim::Tensor::CreateHostTensor(decode_current_length_input_info,
                                                                                   decode_current_length_mem_size,
                                                                                   &decode_current_length);
  decode_input_map.insert(
      std::pair<std::string, tcim::Tensor>(decode_current_length_name, decode_current_length_input_tensor));
  // tokenizer init
  tokenizer = std::make_shared<HmTokenizer>(tokenizerJsonPath,
                                            embeddingWeightPath,
                                            this->embedding_length,
                                            this->prefill_length);

  this->eos_token_id = tokenizer->Encode("<|im_end|>")[0];
}

std::shared_ptr<tcim::Module> HmQwenInfer::GetModule(int model_type)
{
  switch (model_type)
  {
  case 0:
    return this->prefill_module;

  case 1:
    return this->decode_module;

  default:
    return nullptr;
  }
}

std::shared_ptr<HmTokenizer> HmQwenInfer::GetTokenizer()
{
  return this->tokenizer;
}

int HmQwenInfer::get_nblocks()
{
  int count = 0;
  static const std::regex pattern(R"(^model_layers_(\d+)_self_attn_kcache_input$)");
  int input_num = prefill_module->GetInputNum();
  for (int idx = 0; idx < input_num; idx++)
  {
    std::string input_name = prefill_module->GetInputName(idx);
    if (std::regex_match(input_name, pattern))
    {
      ++count;
    }
  }
  return count;
}

void HmQwenInfer::DebugModelInfo(tcim::Module &module, const std::string &modelName)
{
  std::cout << "Model Name : " << modelName << std::endl;
  int input_num = module.GetInputNum();
  for (int idx = 0; idx < input_num; idx++)
  {
    auto input_name = module.GetInputName(idx);
    auto input_info = module.GetInputInfo(input_name).AsContiguous();
    std::cout << "Input[" << input_name << "] " << input_info << std::endl;
  }

  int output_num = module.GetOutputNum();
  for (int idx = 0; idx < output_num; idx++)
  {
    auto output_name = module.GetOutputName(idx);
    auto output_info = module.GetOutputInfo(output_name).AsContiguous();
    std::cout << "Output[" << output_name << "] " << output_info << std::endl;
  }
}

HmQwenInfer::~HmQwenInfer()
{
  prefill_module.reset();
  decode_module.reset();
  tokenizer.reset();
}

void HmQwenInfer::PrefillSetInputDatas(void *data, int32_t valid_length, int32_t current_length)
{
  for (int idx = 0; idx < 3; idx++)
  {
    auto input_name = prefill_module->GetInputName(idx);
    auto input_info = prefill_module->GetInputInfo(input_name).AsContiguous();

    tcim::Tensor input_tensor;
    size_t mem_size = 0;
    if (input_name == "input_1")
    {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
    }
    else if (input_name == "valid_length")
    {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, &valid_length);
    }
    else if (input_name == "current_length")
    {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, &current_length);
    }
    else
    {
      break;
    }

    if (prefill_input_map.find(input_name) != prefill_input_map.end())
    {
      prefill_input_map.at(input_name) = input_tensor;
    }
    else
    {
      prefill_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
  }

  for (const auto &input : prefill_input_map)
  {
    prefill_module->SetInput(input.first, input.second);
  }
}

void HmQwenInfer::PrefillInfer()
{
  prefill_module->Run();
  prefill_module->Sync();
}

void HmQwenInfer::PrefillGetOutputDatas(std::vector<int32_t> &ids)
{
  int output_num = prefill_module->GetOutputNum();

  auto output_name = prefill_module->GetOutputName(0);
  auto output_info = prefill_module->GetOutputInfo(output_name)
                         .AsContiguous();

  auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
  prefill_output_map.insert(
      std::pair<std::string, tcim::Tensor>(output_name, output_tensor));

  auto output = *prefill_output_map.begin();
  output_tensor = prefill_module->GetOutput(output.first);
  output_tensor.CastTo(output.second);

  void *prefill_outData = output_tensor.Data();
  ids.emplace_back(eigen_argmax_half(reinterpret_cast<half *>(prefill_outData), argmax_dim_len));
}

void HmQwenInfer::DecodeSetInputDatas(void *data, int32_t context_length)
{
  for (int idx = 0; idx < 2; idx++)
  {
    auto input_name = decode_module->GetInputName(idx);
    auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();

    tcim::Tensor input_tensor;
    size_t mem_size = 0;
    if (idx == 0)
    {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
    }
    else if (idx == 1)
    {
      mem_size = input_info.MemSize();
      input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, &context_length);
    }

    if (decode_input_map.find(input_name) != decode_input_map.end())
    {
      decode_input_map.at(input_name) = input_tensor;
    }
    else
    {
      decode_input_map.insert(
          std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
    }
  }

  for (const auto &input : decode_input_map)
  {
    decode_module->SetInput(input.first, input.second);
  }
}
void HmQwenInfer::DecodeInfer()
{
  decode_module->Run();
  decode_module->Sync();
}

void HmQwenInfer::DecodeGetOutputDatas(std::vector<int32_t> &ids)
{
#if 1
  int output_num = decode_module->GetOutputNum();
  auto output_name = decode_module->GetOutputName(0);
  auto output_info = decode_module->GetOutputInfo(output_name)
                         .AsContiguous();

  auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);

  if (decode_output_map.find(output_name) != decode_output_map.end())
  {
    decode_output_map.at(output_name) = output_tensor;
  }
  else
  {
    decode_output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }

  auto output = *decode_output_map.begin();
  output_tensor = decode_module->GetOutput(output.first);
  output_tensor.CastTo(output.second);

  void *decode_outData = output_tensor.Data();
  ids.emplace_back(eigen_argmax_half(reinterpret_cast<half *>(decode_outData), argmax_dim_len));
#else
  auto output_name = decode_module->GetOutputName(0);
  if (decode_output_map.find(output_name) != decode_output_map.end())
  {
    ;
  }
  else
  {
    auto output_info = decode_module->GetOutputInfo(output_name)
                           .AsContiguous();

    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    decode_output_map.insert(
        std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
  }

  decode_output_map.at(output_name) = decode_module->GetOutput(output_name);

  void *decode_outData = decode_output_map.at(output_name).Data();

  ids.emplace_back(eigen_argmax_half(reinterpret_cast<half *>(decode_outData), argmax_dim_len));
#endif
}

void HmQwenInfer::chat(const std::string &msg)
{
  std::cout << "Question : " << msg << std::endl;
  auto t_start = std::chrono::high_resolution_clock::now();
  PerfInfos llm_perf_datas;
  memset(&llm_perf_datas, 0, sizeof(PerfInfos));
  auto t_embed_start = std::chrono::high_resolution_clock::now();
  auto t_embed_end = std::chrono::high_resolution_clock::now();
  auto t_prefill_end = std::chrono::high_resolution_clock::now();
  auto t_decode_start = std::chrono::high_resolution_clock::now();
  auto t_decode_end = std::chrono::high_resolution_clock::now();
  auto t_prefill_start = std::chrono::high_resolution_clock::now();
  std::vector<Message>
      msgs = {
          {"system", "You are a helpful assistant."},
          {"user", msg}};
  // embedding
  std::string rendered = tokenizer->ApplyChatTemplate(msgs, true, false);
  std::vector<int> all_input_ids = tokenizer->Encode(rendered);
  int32_t input_echo_len = all_input_ids.size();
  if (input_echo_len > context_max_length)
  {
    std::cerr << "Question long than " << context_max_length << ", please shorten it !" << std::endl;
    return;
  }

  int prefill_loop_round = std::ceil((float)input_echo_len / (float)prefill_length);
  int32_t valid_length = 0, current_length = 0;
  half *input_datas = nullptr;
  for (int round = 0; round < prefill_loop_round; round++)
  {
    valid_length = round * prefill_length;
    std::vector<int> input_ids;
    if (round == prefill_loop_round - 1)
    {
      current_length = input_echo_len - round * prefill_length;
      input_ids.reserve(current_length);
      input_ids.assign(all_input_ids.end() - current_length, all_input_ids.end());
    }
    else
    {
      current_length = prefill_length;
      input_ids.reserve(current_length);
      input_ids.assign(all_input_ids.begin() + round * prefill_length,
                       all_input_ids.begin() + (round + 1) * prefill_length);
    }

    t_embed_start = std::chrono::high_resolution_clock::now();
    input_datas = tokenizer->EmbeddingTokens(input_ids);
    t_embed_end = std::chrono::high_resolution_clock::now();
    llm_perf_datas.embedding_time += std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start).count();
    PrefillSetInputDatas(input_datas, valid_length, current_length);
    PrefillInfer();
  }
  std::vector<int> ids;
  PrefillGetOutputDatas(ids);

  std::vector<int> chat_history_ids = all_input_ids;
  std::string prefill_response = tokenizer->Decode(ids);
  t_prefill_end = std::chrono::high_resolution_clock::now();
  llm_perf_datas.prefill_time += std::chrono::duration<float, std::milli>(t_prefill_end - t_prefill_start).count();
  chat_history_ids.emplace_back(ids[0]);
  t_embed_start = std::chrono::high_resolution_clock::now();
  input_datas = tokenizer->EmbeddingTokens(ids);
  t_embed_end = std::chrono::high_resolution_clock::now();
  llm_perf_datas.embedding_time += std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start).count();

  std::string all_response = prefill_response;
  int context_length = input_echo_len;

  std::cout << "Response : " << prefill_response;
  uint32_t decode_count = 0,
           skip_tokens = 0, slide_len = 10;
  std::vector<int> slide_window_ids(chat_history_ids.end() - slide_len, chat_history_ids.end());
  std::string last_response = tokenizer->Decode(slide_window_ids);
  std::string decode_response;

  t_decode_start = std::chrono::high_resolution_clock::now();
  do
  {
    if (context_length > context_max_length)
    {
      break;
    }
    DecodeSetInputDatas(reinterpret_cast<void *>(input_datas),
                        reinterpret_cast<int32_t>(context_length));
    DecodeInfer();
    ids.clear();
    DecodeGetOutputDatas(ids);
    decode_count++;
    if (ids[0] == eos_token_id)
    {
      std::cout << decode_response << std::endl;
      all_response += decode_response;
      break;
    }

    chat_history_ids.emplace_back(ids[0]);
    int substart = utf8_len(last_response);
    std::vector<int> decode_window_ids(chat_history_ids.end() - slide_len - skip_tokens - 1, chat_history_ids.end());
    std::string tmp_response = tokenizer->Decode(decode_window_ids);
    std::u32string udecode_response = utf8_to_u32(tmp_response).substr(substart);
    decode_response = u32_to_utf8(udecode_response);
    if (decode_response != "" && is_valid_char(udecode_response.back()))
    {
      std::cout << decode_response;
      all_response += decode_response;
      std::vector<int> cur_slide_win(chat_history_ids.end() - slide_len, chat_history_ids.end());
      last_response = tokenizer->Decode(cur_slide_win);
      skip_tokens = 0;
    }
    else
    {
      skip_tokens += 1;
    }
    t_embed_start = std::chrono::high_resolution_clock::now();
    input_datas = tokenizer->EmbeddingTokens(ids);
    t_embed_end = std::chrono::high_resolution_clock::now();
    llm_perf_datas.embedding_time += std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start).count();
    context_length = context_length + 1;
  } while (true);
  t_decode_end = std::chrono::high_resolution_clock::now();
  llm_perf_datas.decode_time += std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start).count();
  auto t_end = std::chrono::high_resolution_clock::now();
  float t_total = std::chrono::duration<float, std::milli>(t_end - t_start).count();
  // perf information
  std::cout << std::fixed << std::setprecision(3)
            << "[SUCCESS] Total Input: " << input_echo_len << " tokens, Output "
            << decode_count + 1 << " tokens, Prefill Cost "
            << llm_perf_datas.prefill_time << " ms, Decode Cost "
            << llm_perf_datas.decode_time << " ms\n";

  std::cout << "[SUCCESS] Prefill Speed: "
            << std::setprecision(2) << input_echo_len / (llm_perf_datas.prefill_time * 0.001f)
            << " tokens/s; Decode Speed: "
            << (decode_count) / (llm_perf_datas.decode_time * 0.001f) << " tokens/s\n";

  std::cout << std::setprecision(3)
            << "[SUCCESS] TTFT (Time to First Token): "
            << llm_perf_datas.prefill_time << " ms\n";

  std::cout << "[SUCCESS] TPOT (Time Per Output Token): "
            << llm_perf_datas.decode_time / (decode_count) << " ms/token\n";

  std::cout << std::setprecision(3)
            << "[SUCCESS] E2E Latency (End-to-End Latency): "
            << t_total * 0.001f << " seconds\n";

  std::cout << std::setprecision(2)
            << "[SUCCESS] E2E TPS (End-to-End Tokens Per Second): "
            << (decode_count + 1) / (t_total * 0.001f) << " tokens/s\n";

  std::cout << std::setprecision(2)
            << "[SUCCESS] All Embedding takes " << llm_perf_datas.embedding_time << " ms.\n";
}