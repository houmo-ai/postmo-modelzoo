/*
 * HmQwenInfer.h - C++ inference interface for HmQwen language model
 *
 * Copyright (c) 2025 HOUMOAI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "HmQwenInfer.h"

/**
 * @brief Constructor for HmQwenInfer class. Initializes prefill/decode modules, tokenizer, and model configurations.
 *
 * @param prefillModelPath Path to prefill model file
 * @param decodeModelPath Path to decode model file
 * @param tokenizerJsonPath Path to tokenizer JSON file
 * @param embeddingWeightPath Path to embedding weight file
 */
HmQwenInfer::HmQwenInfer(const std::string &prefillModelPath,
                         const std::string &decodeModelPath,
                         const std::string &tokenizerJsonPath,
                         const std::string &embeddingWeightPath) {
    this->prefillModelPath = prefillModelPath;
    this->decodeModelPath = decodeModelPath;

    // Create weight manager for model weight management
    weight_manager = tcim::Module::WeightManager::CreateWeightManager(0);
    auto option_prefill = tcim::Module::Option(weight_manager);
    auto option_decode = tcim::Module::Option(weight_manager);

    // Initialize prefill module
    prefill_module = std::make_shared<tcim::Module>();
    prefill_module->LoadModel(prefillModelPath, option_prefill);

    // Get number of transformer blocks in the model
    int n_blocks = GetnBlocks();

    // Generate dummy tensor names for k-cache and v-cache inputs
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

    // Set dummy tensors for decode module and initialize it
    option_decode.SetDummyTensors(dummy_names);
    decode_module = std::make_shared<tcim::Module>();
    decode_module->LoadModel(decodeModelPath, option_decode);

    // Extract model configuration parameters from input/output tensor shapes
    this->prefill_length = prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[1];
    this->embedding_length = prefill_module->GetInputInfo(prefill_module->GetInputName(0)).Shape()[2];
    this->context_max_length = prefill_module->GetInputInfo(prefill_module->GetInputName(3)).Shape()[2];
    this->batch = decode_module->GetInputInfo(decode_module->GetInputName(0)).Shape()[0];
    this->argmax_dim_len = decode_module->GetOutputInfo(decode_module->GetOutputName(0)).Shape()[2];

    // Configure decode module input cache tensors from prefill module inputs
    for (int idx = 3; idx < 2 * n_blocks + 3; idx++) {
        const std::string input_name = prefill_module->GetInputName(idx);
        auto cache = prefill_module->GetInput(input_name);
        decode_input_map.insert(std::pair<std::string, tcim::Tensor>(
            decode_module->GetInputName(idx), cache));
    }

    // Set up decode current_length input tensor
    const std::string decode_current_length_name = decode_module->GetInputName(2);
    const tcim::TensorInfo decode_current_length_input_info = decode_module->GetInputInfo(decode_current_length_name).AsContiguous();
    size_t decode_current_length_mem_size = decode_current_length_input_info.MemSize();
    tcim::Tensor decode_current_length_input_tensor = tcim::Tensor::CreateHostTensor(decode_current_length_input_info, decode_current_length_mem_size, &decode_current_length);
    decode_input_map.insert(std::pair<std::string, tcim::Tensor>(decode_current_length_name, decode_current_length_input_tensor));

    // Initialize tokenizer
    tokenizer = std::make_shared<HmTokenizer>(tokenizerJsonPath, embeddingWeightPath, this->embedding_length, this->prefill_length);

    // Set end-of-sequence token ID
    this->eos_token_id = tokenizer->Encode("<|im_end|>")[0];
}

/**
 * @brief Get the appropriate module (prefill or decode) based on model type.
 *
 * @param model_type 0 for prefill module, 1 for decode module
 * @return std::shared_ptr<tcim::Module> Pointer to the requested module, or nullptr if invalid model_type
 */
std::shared_ptr<tcim::Module> HmQwenInfer::GetModule(int model_type) {
    switch (model_type) {
    case 0:
        return this->prefill_module;

    case 1:
        return this->decode_module;

    default:
        return nullptr;
    }
}

/**
 * @brief Get the tokenizer instance.
 *
 * @return std::shared_ptr<HmTokenizer> Pointer to the tokenizer
 */
std::shared_ptr<HmTokenizer> HmQwenInfer::GetTokenizer() {
    return this->tokenizer;
}

/**
 * @brief Calculate the number of transformer blocks in the model by counting cache input tensors.
 *
 * @return int Number of transformer blocks
 */
int HmQwenInfer::GetnBlocks() {
    int count = 0;
    static const std::regex pattern(R"(^model_layers_(\d+)_self_attn_kcache_input$)");
    int input_num = prefill_module->GetInputNum();

    for (int idx = 0; idx < input_num; idx++) {
        std::string input_name = prefill_module->GetInputName(idx);
        if (std::regex_match(input_name, pattern)) {
            ++count;
        }
    }

    return count;
}

/**
 * @brief Debug function to print model input/output tensor information.
 *
 * @param module Reference to the tcim Module
 * @param modelName Name of the model to identify in output
 */
void HmQwenInfer::DebugModelInfo(tcim::Module &module, const std::string &modelName) {
    std::cout << "Model Name : " << modelName << std::endl;

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

/**
 * @brief Destructor for HmQwenInfer class. Releases module and tokenizer resources.
 */
HmQwenInfer::~HmQwenInfer() {
    prefill_module.reset();
    decode_module.reset();
    tokenizer.reset();
}

/**
 * @brief Set input data for prefill inference.
 *
 * @param data Pointer to input embedding data
 * @param valid_length Number of valid tokens in the context
 * @param current_length Number of tokens in the current prefill batch
 */
void HmQwenInfer::PrefillSetInputDatas(void *data, int32_t valid_length, int32_t current_length) {
    prefill_input_map.clear();

    for (int idx = 0; idx < 3; idx++) {
        auto input_name = prefill_module->GetInputName(idx);
        auto input_info = prefill_module->GetInputInfo(input_name).AsContiguous();

        tcim::Tensor input_tensor;
        size_t mem_size = 0;

        // Create host tensor for each input type
        if (idx == 0) {  // Embedding data
            mem_size = input_info.MemSize();
            input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
        } else if (idx == 1) {  // Valid length
            mem_size = input_info.MemSize();
            input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, &valid_length);
        } else if (idx == 2) {  // Current length
            mem_size = input_info.MemSize();
            input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, &current_length);
        }

        // Update or insert input tensor into map
        if (prefill_input_map.find(input_name) != prefill_input_map.end()) {
            prefill_input_map.at(input_name) = input_tensor;
        } else {
            prefill_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }
    }

    // Set all inputs for the prefill module
    for (const auto &input : prefill_input_map) {
        prefill_module->SetInput(input.first, input.second);
    }
}

/**
 * @brief Run prefill inference and synchronize.
 */
void HmQwenInfer::PrefillInfer() {
    prefill_module->Run();
    prefill_module->Sync();
}

/**
 * @brief Get output tokens from prefill inference.
 *
 * @param ids Output vector to store generated token IDs
 */
void HmQwenInfer::PrefillGetOutputDatas(std::vector<int32_t> &ids) {
    auto output_name = prefill_module->GetOutputName(0);
    auto output_info = prefill_module->GetOutputInfo(output_name).AsContiguous();

    // Create host tensor for output
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    prefill_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));

    // Retrieve and cast output tensor
    auto output = *prefill_output_map.begin();
    output_tensor = prefill_module->GetOutput(output.first);
    output_tensor.CastTo(output.second);

    // Apply argmax to get the predicted token
    void *prefill_outData = output_tensor.Data();
    ids.emplace_back(eigen_argmax<tensor_type>(static_cast<tensor_type *>(prefill_outData), argmax_dim_len));
}

/**
 * @brief Set input data for decode inference.
 *
 * @param data Pointer to input embedding data (single token)
 * @param context_length Current context length
 */
void HmQwenInfer::DecodeSetInputDatas(void *data, int32_t context_length) {
    for (int idx = 0; idx < 2; idx++) {
        auto input_name = decode_module->GetInputName(idx);
        auto input_info = decode_module->GetInputInfo(input_name).AsContiguous();

        tcim::Tensor input_tensor;
        size_t mem_size = 0;

        // Create host tensor for each input type
        if (idx == 0) {  // Embedding data
            mem_size = input_info.MemSize();
            input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, data);
        } else if (idx == 1) {  // Context length
            mem_size = input_info.MemSize();
            input_tensor = tcim::Tensor::CreateHostTensor(input_info, mem_size, &context_length);
        }

        // Update or insert input tensor into map
        if (decode_input_map.find(input_name) != decode_input_map.end()) {
            decode_input_map.at(input_name) = input_tensor;
        } else {
            decode_input_map.insert(std::pair<std::string, tcim::Tensor>(input_name, input_tensor));
        }
    }

    // Set all inputs for the decode module
    for (const auto &input : decode_input_map) {
        decode_module->SetInput(input.first, input.second);
    }
}

/**
 * @brief Run decode inference and synchronize.
 */
void HmQwenInfer::DecodeInfer() {
    decode_module->Run();
    decode_module->Sync();
}

/**
 * @brief Get output tokens from decode inference.
 *
 * @param ids Output vector to store generated token IDs
 */
void HmQwenInfer::DecodeGetOutputDatas(std::vector<int32_t> &ids) {
    auto output_name = decode_module->GetOutputName(0);
    auto output_info = decode_module->GetOutputInfo(output_name).AsContiguous();

    // Create host tensor for output
    auto output_tensor = tcim::Tensor::CreateHostTensor(output_info);
    if (decode_output_map.find(output_name) != decode_output_map.end()) {
        decode_output_map.at(output_name) = output_tensor;
    } else {
        decode_output_map.insert(std::pair<std::string, tcim::Tensor>(output_name, output_tensor));
    }

    // Retrieve and cast output tensor
    auto output = *decode_output_map.begin();
    output_tensor = decode_module->GetOutput(output.first);
    output_tensor.CastTo(output.second);

    // Apply argmax to get the predicted token
    void *decode_outData = output_tensor.Data();
    ids.emplace_back(eigen_argmax<tensor_type>(static_cast<tensor_type *>(decode_outData), argmax_dim_len));
}

/**
 * @brief Performs the main chat inference function, processing user input and generating AI responses
 *
 * This function implements the complete LLM inference process, including prefill and decode phases,
 * with performance monitoring and statistics
 *
 * @param msg The user input message string for the chat
 * @return void No return value, directly outputs results to console
 */
void HmQwenInfer::Chat(const std::string &msg) {
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
    std::vector<Message> msgs = {{"system", "You are a helpful assistant."},
                                 {"user", msg}};
    // Process the input through tokenizer chat template
    std::string rendered = tokenizer->ApplyChatTemplate(msgs, true, false);
    std::vector<int> all_input_ids = tokenizer->Encode(rendered);
    int32_t input_echo_len = all_input_ids.size();
    if (input_echo_len > context_max_length) {
        std::cerr << "Question longer than " << context_max_length
                  << ", please shorten it !" << std::endl;
        return;
    }

    // Calculate number of prefill rounds to handle long inputs in chunks
    int prefill_loop_round =
        std::ceil((float)input_echo_len / (float)prefill_length);
    int32_t valid_length = 0, current_length = 0;
    tensor_type *input_datas = nullptr;
    for (int round = 0; round < prefill_loop_round; round++) {
        valid_length = round * prefill_length;
        std::vector<int> input_ids;
        if (round == prefill_loop_round - 1) {
            current_length = input_echo_len - round * prefill_length;
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
        input_datas = tokenizer->EmbeddingTokens(input_ids);
        t_embed_end = std::chrono::high_resolution_clock::now();
        llm_perf_datas.embedding_time +=
            std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
                .count();
        PrefillSetInputDatas(input_datas, valid_length, current_length);
        PrefillInfer();
    }
    std::vector<int> ids;
    PrefillGetOutputDatas(ids);

    std::vector<int> chat_history_ids = all_input_ids;
    std::string prefill_response = tokenizer->Decode(ids);
    t_prefill_end = std::chrono::high_resolution_clock::now();
    llm_perf_datas.prefill_time +=
        std::chrono::duration<float, std::milli>(t_prefill_end - t_prefill_start)
            .count();
    chat_history_ids.emplace_back(ids[0]);
    t_embed_start = std::chrono::high_resolution_clock::now();
    input_datas = tokenizer->EmbeddingTokens(ids);
    t_embed_end = std::chrono::high_resolution_clock::now();
    llm_perf_datas.embedding_time +=
        std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
            .count();

    std::string all_response = prefill_response;
    int context_length = input_echo_len;

    std::cout << "Response : " << prefill_response;
    uint32_t decode_count = 0, skip_tokens = 0, slide_len = 10;
    std::vector<int> slide_window_ids(chat_history_ids.end() - slide_len,
                                      chat_history_ids.end());
    std::string last_response = tokenizer->Decode(slide_window_ids);
    std::string decode_response;

    t_decode_start = std::chrono::high_resolution_clock::now();
    do {
        if (context_length > context_max_length) {
            break;
        }
        DecodeSetInputDatas(static_cast<void *>(input_datas),
                            static_cast<int32_t>(context_length));
        DecodeInfer();
        ids.clear();
        DecodeGetOutputDatas(ids);
        decode_count++;
        if (ids[0] == eos_token_id) {
            std::cout << decode_response << std::endl;
            all_response += decode_response;
            break;
        }

        chat_history_ids.emplace_back(ids[0]);
        int substart = utf8_len(last_response);
        std::vector<int> decode_window_ids(
            chat_history_ids.end() - slide_len - skip_tokens - 1,
            chat_history_ids.end());
        std::string tmp_response = tokenizer->Decode(decode_window_ids);
        std::u32string udecode_response =
            utf8_to_u32(tmp_response).substr(substart);
        decode_response = u32_to_utf8(udecode_response);
        if (decode_response != "" && is_valid_char(udecode_response.back())) {
            std::cout << decode_response << std::flush;
            all_response += decode_response;
            std::vector<int> cur_slide_win(chat_history_ids.end() - slide_len,
                                           chat_history_ids.end());
            last_response = tokenizer->Decode(cur_slide_win);
            skip_tokens = 0;
        } else {
            skip_tokens += 1;
        }
        t_embed_start = std::chrono::high_resolution_clock::now();
        input_datas = tokenizer->EmbeddingTokens(ids);
        t_embed_end = std::chrono::high_resolution_clock::now();
        llm_perf_datas.embedding_time +=
            std::chrono::duration<float, std::milli>(t_embed_end - t_embed_start)
                .count();
        context_length = context_length + 1;
    } while (true);
    t_decode_end = std::chrono::high_resolution_clock::now();
    llm_perf_datas.decode_time +=
        std::chrono::duration<float, std::milli>(t_decode_end - t_decode_start)
            .count();
    auto t_end = std::chrono::high_resolution_clock::now();
    float t_total =
        std::chrono::duration<float, std::milli>(t_end - t_start).count();
    // Output performance statistics including prefill time, decode time, throughput and other key metrics
    std::cout << std::fixed << std::setprecision(3)
              << "[SUCCESS] Total Input: " << input_echo_len << " tokens, Output "
              << decode_count + 1 << " tokens, Prefill Cost "
              << llm_perf_datas.prefill_time << " ms, Decode Cost "
              << llm_perf_datas.decode_time << " ms\n";

    std::cout << "[SUCCESS] Prefill Speed: " << std::setprecision(2)
              << input_echo_len / (llm_perf_datas.prefill_time * 0.001f)
              << " tokens/s; Decode Speed: "
              << (decode_count) / (llm_perf_datas.decode_time * 0.001f)
              << " tokens/s\n";

    std::cout << std::setprecision(3) << "[SUCCESS] TTFT (Time to First Token): "
              << llm_perf_datas.prefill_time << " ms\n";

    std::cout << "[SUCCESS] TPOT (Time Per Output Token): "
              << llm_perf_datas.decode_time / (decode_count) << " ms/token\n";

    std::cout << std::setprecision(3)
              << "[SUCCESS] E2E Latency (End-to-End Latency): "
              << t_total * 0.001f << " seconds\n";

    std::cout << std::setprecision(2)
              << "[SUCCESS] E2E TPS (End-to-End Tokens Per Second): "
              << (decode_count + 1) / (t_total * 0.001f) << " tokens/s\n";

    std::cout << std::setprecision(2) << "[SUCCESS] All Embedding takes "
              << llm_perf_datas.embedding_time << " ms.\n";
}