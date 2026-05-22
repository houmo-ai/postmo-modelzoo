/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: HmQwenVLInfer.cc
 * Description:
 *   Main inference implementation for Qwen3-VL model.
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

#include "HmQwenVLInfer.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>

// Model configuration constants
constexpr int IMAGE_TOKEN_ID = 151655;
constexpr int VIDEO_TOKEN_ID = 151656;
constexpr int VISION_START_TOKEN_ID = 151652;
constexpr int VISION_END_TOKEN_ID = 151653;
constexpr int EOS_TOKEN_ID = 151645;

HmQwenVLInfer::HmQwenVLInfer(const std::string &visualModelPath,
                             const std::string &prefillModelPath,
                             const std::string &decodeModelPath,
                             const std::string &tokenizerJsonPath,
                             const std::string &embeddingWeightPath,
                             const SamplingManager &sampling_manager)
    : visual_model_path_(visualModelPath),
      prefill_model_path_(prefillModelPath),
      decode_model_path_(decodeModelPath),
      sampling_manager_(sampling_manager) {
  // Create weight manager
  weight_manager_ = tcim::Module::WeightManager::CreateWeightManager(0);

  // Create options
  auto option_visual = tcim::Module::Option(weight_manager_);
  auto option_prefill = tcim::Module::Option(weight_manager_);
  auto option_decode = tcim::Module::Option(weight_manager_);

  // Load visual model
  Timer vision_load_timer;
  vision_load_timer.start();
  visual_module_ = std::make_shared<tcim::Module>();
  visual_module_->LoadModel(visualModelPath, option_visual);
  vision_load_timer.end();
  perf_info_.vision_model_load_time = vision_load_timer.elapsed_ms();
  std::cout << "Visual model loaded" << std::endl;

  // Load prefill model
  Timer prefill_load_timer;
  prefill_load_timer.start();
  prefill_module_ = std::make_shared<tcim::Module>();
  prefill_module_->LoadModel(prefillModelPath, option_prefill);
  prefill_load_timer.end();
  perf_info_.prefill_model_load_time = prefill_load_timer.elapsed_ms();
  std::cout << "Prefill model loaded" << std::endl;

  // Get number of blocks and generate dummy names
  int n_blocks = GetNBlocks();
  for (int i = 0; i < n_blocks; i++) {
    std::stringstream ss;
    ss << "model_layers_" << i << "_self_attn_kcache_input";
    dummy_names_.emplace_back(ss.str());
  }
  for (int i = 0; i < n_blocks; i++) {
    std::stringstream ss;
    ss << "model_layers_" << i << "_self_attn_vcache_input";
    dummy_names_.emplace_back(ss.str());
  }

  // Set dummy tensors for decode module
  option_decode.SetDummyTensors(dummy_names_);

  // Load decode model
  Timer decode_load_timer;
  decode_load_timer.start();
  decode_module_ = std::make_shared<tcim::Module>();
  decode_module_->LoadModel(decodeModelPath, option_decode);
  decode_load_timer.end();
  perf_info_.decode_model_load_time = decode_load_timer.elapsed_ms();
  std::cout << "Decode model loaded" << std::endl;

  // Get attention index start
  attn_idx_start_ = GetAttnIdxStart();

  // Get model dimensions from tensor shapes
  prefill_length_ =
      prefill_module_->GetInputInfo(prefill_module_->GetInputName(0))
          .Shape()[1];
  embedding_length_ =
      prefill_module_->GetInputInfo(prefill_module_->GetInputName(0))
          .Shape()[2];
  context_max_length_ =
      prefill_module_
          ->GetInputInfo(prefill_module_->GetInputName(attn_idx_start_))
          .Shape()[2];
  batch_ =
      decode_module_->GetInputInfo(decode_module_->GetInputName(0)).Shape()[0];
  argmax_dim_len_ =
      decode_module_->GetOutputInfo(decode_module_->GetOutputName(0))
          .Shape()[2];
  vision_input_nums_ = visual_module_->GetInputNum();

  // Get vision model input shape (for image dimensions)
  auto vit_input_shape =
      visual_module_->GetInputInfo(visual_module_->GetInputName(0)).Shape();
  std::cout << "VIT input shape: ";
  for (size_t i = 0; i < vit_input_shape.size(); i++) {
    std::cout << vit_input_shape[i] << " ";
  }
  std::cout << std::endl;
  // Shape format: [batch, channels, temporal, height, width] (5D for
  // video/image) For TCIM: typically [1, 3, T, H, W]
  if (vit_input_shape.size() >= 5) {
    config_.image_size_h = vit_input_shape[3];  // Height
    config_.image_size_w = vit_input_shape[4];  // Width
  } else if (vit_input_shape.size() >= 4) {
    config_.image_size_h = vit_input_shape[2];  // Height
    config_.image_size_w = vit_input_shape[3];  // Width
  }
  std::cout << "Image dimensions: " << config_.image_size_w << "x"
            << config_.image_size_h << std::endl;

  // Initialize components
  tokenizer_ = std::make_shared<HmQwenVLTokenizer>(
      tokenizerJsonPath, embeddingWeightPath, embedding_length_,
      prefill_length_);
  image_processor_ = std::make_shared<HmImageProcessor>(
      config_.image_size_w, config_.image_size_h, true);

  // Initialize input tensors
  InitVisualInputs();
  InitPrefillInputs();
  InitDecodeInputs();

  // Setup KV cache sharing between prefill and decode
  for (int idx = attn_idx_start_; idx < 2 * n_blocks + attn_idx_start_; idx++) {
    const std::string cache_name = prefill_module_->GetInputName(idx);
    auto cache = prefill_module_->GetDevInput(cache_name);
    decode_module_->SetDevInput(decode_module_->GetInputName(idx), cache);
  }
}

HmQwenVLInfer::~HmQwenVLInfer() {
  visual_module_.reset();
  prefill_module_.reset();
  decode_module_.reset();
  tokenizer_.reset();
  image_processor_.reset();
}

int HmQwenVLInfer::GetNBlocks() {
  int count = 0;
  static const std::regex pattern(
      R"(^model_layers_(\d+)_self_attn_kcache_input$)");
  int input_num = prefill_module_->GetInputNum();

  for (int idx = 0; idx < input_num; idx++) {
    std::string input_name = prefill_module_->GetInputName(idx);
    if (std::regex_match(input_name, pattern)) {
      ++count;
    }
  }
  return count;
}

int HmQwenVLInfer::GetAttnIdxStart() {
  int start = 0;
  static const std::regex pattern(
      R"(^model_layers_(\d+)_self_attn_kcache_input$)");
  int input_num = prefill_module_->GetInputNum();

  for (int idx = 0; idx < input_num; idx++) {
    std::string input_name = prefill_module_->GetInputName(idx);
    if (std::regex_match(input_name, pattern)) {
      start = idx;
      break;
    }
  }
  return start;
}

void HmQwenVLInfer::InitVisualInputs() {
  visual_input_map_.clear();
  for (int idx = 0; idx < vision_input_nums_; idx++) {
    auto input_name = visual_module_->GetInputName(idx);
    auto input_info = visual_module_->GetInputInfo(input_name).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    visual_input_map_[input_name] = input_tensor;
  }
}

void HmQwenVLInfer::InitPrefillInputs() {
  prefill_input_map_.clear();
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    auto input_name = prefill_module_->GetInputName(idx);
    auto input_info = prefill_module_->GetInputInfo(input_name).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    prefill_input_map_[input_name] = input_tensor;
  }
}

void HmQwenVLInfer::InitDecodeInputs() {
  decode_input_map_.clear();
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    auto input_name = decode_module_->GetInputName(idx);
    auto input_info = decode_module_->GetInputInfo(input_name).AsContiguous();
    auto input_tensor = tcim::Tensor::CreateHostTensor(input_info);
    decode_input_map_[input_name] = input_tensor;
  }
}

void HmQwenVLInfer::DebugModelInfo(tcim::Module &module,
                                   const std::string &modelName) {
  std::cout << "Model Name: " << modelName << std::endl;

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

void HmQwenVLInfer::VisionSetInput(
    const std::vector<half_float::half> &visual_data) {
  auto input_name = visual_module_->GetInputName(0);
  auto input_tensor = visual_input_map_[input_name];
  input_tensor.Buffer().CopyFromHost(
      visual_data.data(), visual_data.size() * sizeof(half_float::half));
  visual_module_->SetInput(input_name, input_tensor);
}

void HmQwenVLInfer::VisionInfer() {
  visual_module_->Run();
  visual_module_->Sync();
}

std::tuple<std::vector<half_float::half>, std::vector<half_float::half>,
           std::vector<half_float::half>, std::vector<half_float::half>>
HmQwenVLInfer::VisionGetOutputs() {
  // Get all 4 outputs from vision model
  auto get_output_data =
      [this](int output_idx) -> std::vector<half_float::half> {
    auto output_name = visual_module_->GetOutputName(output_idx);
    auto dev_output = visual_module_->GetDevOutput(output_name);
    auto host_output = dev_output.ToHost(true);
    size_t num_elements =
        host_output.Buffer().Size() / sizeof(half_float::half);
    std::vector<half_float::half> data(num_elements);
    std::memcpy(data.data(), host_output.Buffer().Data(),
                host_output.Buffer().Size());
    return data;
  };

  return {
      get_output_data(0),  // image_features
      get_output_data(1),  // deepstack_image_feature_0
      get_output_data(2),  // deepstack_image_feature_1
      get_output_data(3)   // deepstack_image_feature_2
  };
}

void HmQwenVLInfer::PrefillSetInputDatas(
    const std::vector<half_float::half> &inputs_embeds,
    const std::vector<int32_t> &time_position_ids,
    const std::vector<int32_t> &height_position_ids,
    const std::vector<int32_t> &width_position_ids, int32_t valid_length,
    int32_t current_length,
    const std::vector<half_float::half> &deepstack_embed_0,
    const std::vector<half_float::half> &deepstack_embed_1,
    const std::vector<half_float::half> &deepstack_embed_2) {
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    auto name = prefill_module_->GetInputName(idx);
    auto tensor = prefill_input_map_[name];
    size_t memSize = tensor.MemSize();

    if (idx == 0) {  // inputs_embeds
      tensor.Buffer().CopyFromHost(inputs_embeds.data(), memSize);
    } else if (idx == 1) {  // time_position_ids
      tensor.Buffer().CopyFromHost(time_position_ids.data(),
                                   time_position_ids.size() * sizeof(int32_t));
    } else if (idx == 2) {  // height_position_ids
      tensor.Buffer().CopyFromHost(
          height_position_ids.data(),
          height_position_ids.size() * sizeof(int32_t));
    } else if (idx == 3) {  // width_position_ids
      tensor.Buffer().CopyFromHost(width_position_ids.data(),
                                   width_position_ids.size() * sizeof(int32_t));
    } else if (idx == 4) {  // valid_length (past_seq_length in Python)
      tensor.Buffer().CopyFromHost(&valid_length, sizeof(int32_t));
    } else if (idx == 5) {  // current_length
      tensor.Buffer().CopyFromHost(&current_length, sizeof(int32_t));
    } else if (idx == 6) {  // deepstack_image_embed_0
      tensor.Buffer().CopyFromHost(deepstack_embed_0.data(), memSize);
    } else if (idx == 7) {  // deepstack_image_embed_1
      tensor.Buffer().CopyFromHost(deepstack_embed_1.data(), memSize);
    } else if (idx == 8) {  // deepstack_image_embed_2
      tensor.Buffer().CopyFromHost(deepstack_embed_2.data(), memSize);
    }

    prefill_module_->SetInput(name, tensor);
  }
}

void HmQwenVLInfer::PrefillInfer() {
  prefill_module_->Run();
  prefill_module_->Sync();
}

int HmQwenVLInfer::PrefillGetOutputDatas() {
  auto output_name = prefill_module_->GetOutputName(0);
  auto dev_output = prefill_module_->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);

  void *outData = host_output.Buffer().Data();
  int token_id = eigen_argmax<half_float::half>(
      static_cast<half_float::half *>(outData), argmax_dim_len_);
  return token_id;
}

void HmQwenVLInfer::DecodeSetInputDatas(
    const std::vector<half_float::half> &inputs_embeds,
    const std::vector<int32_t> &time_position_ids,
    const std::vector<int32_t> &height_position_ids,
    const std::vector<int32_t> &width_position_ids, int32_t valid_length,
    const std::vector<half_float::half> &deepstack_embed_0,
    const std::vector<half_float::half> &deepstack_embed_1,
    const std::vector<half_float::half> &deepstack_embed_2) {
  for (int idx = 0; idx < attn_idx_start_; idx++) {
    auto name = decode_module_->GetInputName(idx);
    auto tensor = decode_input_map_[name];

    if (idx == 0) {  // inputs_embeds
      tensor.Buffer().CopyFromHost(
          inputs_embeds.data(),
          inputs_embeds.size() * sizeof(half_float::half));
    } else if (idx == 1) {  // time_position_ids
      tensor.Buffer().CopyFromHost(time_position_ids.data(),
                                   time_position_ids.size() * sizeof(int32_t));
    } else if (idx == 2) {  // height_position_ids
      tensor.Buffer().CopyFromHost(
          height_position_ids.data(),
          height_position_ids.size() * sizeof(int32_t));
    } else if (idx == 3) {  // width_position_ids
      tensor.Buffer().CopyFromHost(width_position_ids.data(),
                                   width_position_ids.size() * sizeof(int32_t));
    } else if (idx == 4) {  // valid_length (past_seq_length)
      tensor.Buffer().CopyFromHost(&valid_length, sizeof(int32_t));
    } else if (idx == 5) {  // current_length (always 1 for decode)
      int32_t current_length = 1;
      tensor.Buffer().CopyFromHost(&current_length, sizeof(int32_t));
    } else if (idx == 6) {  // deepstack_image_embed_0
      tensor.Buffer().CopyFromHost(
          deepstack_embed_0.data(),
          deepstack_embed_0.size() * sizeof(half_float::half));
    } else if (idx == 7) {  // deepstack_image_embed_1
      tensor.Buffer().CopyFromHost(
          deepstack_embed_1.data(),
          deepstack_embed_1.size() * sizeof(half_float::half));
    } else if (idx == 8) {  // deepstack_image_embed_2
      tensor.Buffer().CopyFromHost(
          deepstack_embed_2.data(),
          deepstack_embed_2.size() * sizeof(half_float::half));
    }

    decode_module_->SetInput(name, tensor);
  }
}

void HmQwenVLInfer::DecodeInfer() {
  decode_module_->Run();
  decode_module_->Sync();
}

int HmQwenVLInfer::DecodeGetOutputDatas() {
  auto output_name = decode_module_->GetOutputName(0);
  auto dev_output = decode_module_->GetDevOutput(output_name);
  auto host_output = dev_output.ToHost(true);

  void *outData = host_output.Buffer().Data();
  size_t num_elements = host_output.Buffer().Size() / sizeof(half_float::half);

  // Convert half logits to float for sampling
  std::vector<float> logits(num_elements);
  half_float::half *half_data = static_cast<half_float::half *>(outData);
  for (size_t i = 0; i < num_elements; i++) {
    logits[i] = static_cast<float>(half_data[i]);
  }

  // Apply sampling manager for post-processing
  int token_id =
      sampling_manager_.sample(logits.data(), num_elements, generated_ids_);
  return token_id;
}

bool HmQwenVLInfer::IsValidChar(char32_t cp) {
  return
      // CJK Unified Ideographs
      (cp >= 0x4E00u && cp <= 0x9FFFu) || (cp >= 0x3400u && cp <= 0x4DBFu) ||
      (cp >= 0x20000u && cp <= 0x2A6DFu) ||
      (cp >= 0x2A700u && cp <= 0x2B73Fu) ||
      (cp >= 0x2B740u && cp <= 0x2B81Fu) ||
      (cp >= 0x2B820u && cp <= 0x2CEAFu) || (cp >= 0xF900u && cp <= 0xFAFFu) ||
      (cp >= 0x2F800u && cp <= 0x2FA1Fu) || (cp >= 0x0041u && cp <= 0x005Au) ||
      (cp >= 0x0061u && cp <= 0x007Au);
}

std::string HmQwenVLInfer::Chat(const std::vector<std::string> &image_paths,
                                const std::string &prompt) {
  Timer timer;
  timer.start();
  const float prefill_model_load_time = perf_info_.prefill_model_load_time;
  const float decode_model_load_time = perf_info_.decode_model_load_time;
  const float vision_model_load_time = perf_info_.vision_model_load_time;
  perf_info_ = PerfInfos();
  perf_info_.prefill_model_load_time = prefill_model_load_time;
  perf_info_.decode_model_load_time = decode_model_load_time;
  perf_info_.vision_model_load_time = vision_model_load_time;
  perf_info_.batch_size = batch_ > 0 ? batch_ : 1;
  perf_info_.num_images = static_cast<int>(image_paths.size());

  std::cout << "Question: " << prompt << std::endl;
  if (!image_paths.empty()) {
    std::cout << "Images: ";
    for (const auto &path : image_paths) {
      std::cout << path << " ";
    }
    std::cout << std::endl;
  }

  // Apply chat template
  Timer prefill_tokenize_timer;
  prefill_tokenize_timer.start();
  std::string formatted =
      tokenizer_->ApplyChatTemplate(prompt, image_paths, true);

  // Encode to tokens
  std::vector<int> input_ids = tokenizer_->Encode(formatted);
  prefill_tokenize_timer.end();
  perf_info_.prefill_tokenization_time = prefill_tokenize_timer.elapsed_ms();

  if (input_ids.size() > static_cast<size_t>(context_max_length_)) {
    std::cerr << "Input too long: " << input_ids.size() << " > "
              << context_max_length_ << std::endl;
    return "";
  }

  // Process images and run vision model
  std::vector<half_float::half> image_features;
  std::vector<half_float::half> deepstack_0, deepstack_1, deepstack_2;
  std::vector<std::tuple<int, int, int>>
      image_grid_thw;  // (t, h, w) for each image

  Timer vision_timer;
  vision_timer.start();

  if (!image_paths.empty()) {
    // Load and preprocess images
    Timer vision_preprocess_timer;
    vision_preprocess_timer.start();
    auto images = image_processor_->LoadAndProcessBatch(image_paths);
    vision_preprocess_timer.end();
    perf_info_.vision_preprocess_time += vision_preprocess_timer.elapsed_ms();

    for (const auto &img : images) {
      // Convert to YUV tensor
      auto visual_tensor = image_processor_->ToHalfTensor(img);

      // Run vision model
      Timer vision_set_input_timer;
      vision_set_input_timer.start();
      VisionSetInput(visual_tensor);
      vision_set_input_timer.end();
      perf_info_.vision_set_input_time += vision_set_input_timer.elapsed_ms();

      Timer vision_infer_timer;
      vision_infer_timer.start();
      VisionInfer();
      vision_infer_timer.end();
      perf_info_.vision_infer_time += vision_infer_timer.elapsed_ms();
      // Get outputs
      Timer vision_get_output_timer;
      vision_get_output_timer.start();
      auto [feat, ds0, ds1, ds2] = VisionGetOutputs();
      vision_get_output_timer.end();
      perf_info_.vision_get_output_time += vision_get_output_timer.elapsed_ms();
      // Append to buffers
      image_features.insert(image_features.end(), feat.begin(), feat.end());
      deepstack_0.insert(deepstack_0.end(), ds0.begin(), ds0.end());
      deepstack_1.insert(deepstack_1.end(), ds1.begin(), ds1.end());
      deepstack_2.insert(deepstack_2.end(), ds2.begin(), ds2.end());

      // Calculate image grid (t, h, w) for 3D RoPE
      // Grid is calculated from vit model input size, not from feature
      // dimensions grid_h = image_h // (patch_size * spatial_merge_size),
      // grid_w = image_w // (patch_size * spatial_merge_size)
      int grid_h = config_.image_size_h /
                   (config_.patch_size * config_.spatial_merge_size);
      int grid_w = config_.image_size_w /
                   (config_.patch_size * config_.spatial_merge_size);
      image_grid_thw.push_back({1, grid_h, grid_w});
    }
  }

  vision_timer.end();
  perf_info_.vision_time = vision_timer.elapsed_ms();

  // Expand image pad tokens based on vision features count
  if (!image_paths.empty() && image_features.size() > 0) {
    size_t total_image_tokens = image_features.size() / embedding_length_;

    // Find and expand each <|image_pad|> token (151655)
    std::vector<int> expanded_input_ids;
    size_t image_idx = 0;
    for (size_t i = 0; i < input_ids.size(); i++) {
      if (input_ids[i] == IMAGE_TOKEN_ID && image_idx < image_grid_thw.size()) {
        // Calculate num_image_tokens for this image
        int grid_w = config_.image_size_w /
                     (config_.patch_size * config_.spatial_merge_size);
        int grid_h = config_.image_size_h /
                     (config_.patch_size * config_.spatial_merge_size);
        int num_tokens = grid_w * grid_h;

        // Insert multiple <|image_pad|> tokens
        for (int j = 0; j < num_tokens; j++) {
          expanded_input_ids.push_back(IMAGE_TOKEN_ID);
        }
        image_idx++;
      } else {
        expanded_input_ids.push_back(input_ids[i]);
      }
    }
    input_ids = expanded_input_ids;
  }

  // Prepare inputs for prefill
  perf_info_.input_tokens = input_ids.size();
  int32_t seq_length = input_ids.size();
  int32_t input_seq_length =
      ((seq_length + prefill_length_ - 1) / prefill_length_) * prefill_length_;

  // Pad input_ids if needed
  if (input_seq_length > seq_length) {
    input_ids.resize(input_seq_length, 0);  // Pad with pad_token_id=0
  }

  // Get embeddings for all tokens
  std::vector<half_float::half> inputs_embeds(input_seq_length *
                                              embedding_length_);

  // Compute text embeddings
  Timer prefill_embedding_timer;
  prefill_embedding_timer.start();
  auto text_embeds = tokenizer_->EmbeddingTokens(input_ids);
  prefill_embedding_timer.end();
  perf_info_.prefill_embedding_time = prefill_embedding_timer.elapsed_ms();
  std::memcpy(inputs_embeds.data(), text_embeds,
              input_seq_length * embedding_length_ * sizeof(half_float::half));

  // Scatter image embeddings to image token positions
  size_t image_feature_idx = 0;
  for (size_t i = 0;
       i < input_ids.size() && image_feature_idx < image_features.size(); i++) {
    if (input_ids[i] == IMAGE_TOKEN_ID) {
      size_t feat_offset = 0;
      // Find how many image tokens we have
      size_t num_image_tokens = 0;
      for (size_t j = i; j < input_ids.size() && input_ids[j] == IMAGE_TOKEN_ID;
           j++) {
        num_image_tokens++;
      }

      // Copy vision features to these positions
      for (size_t j = 0;
           j < num_image_tokens && image_feature_idx < image_features.size();
           j++) {
        size_t embed_offset = (i + j) * embedding_length_;
        size_t feat_size = embedding_length_;
        if (image_feature_idx + feat_size <= image_features.size()) {
          std::memcpy(inputs_embeds.data() + embed_offset,
                      image_features.data() + image_feature_idx,
                      feat_size * sizeof(half_float::half));
        }
        image_feature_idx += feat_size;
      }
      i += num_image_tokens - 1;  // Skip processed tokens
    }
  }

  // Calculate 3D RoPE position indices
  std::vector<int32_t> time_position_ids(input_seq_length, 0);
  std::vector<int32_t> height_position_ids(input_seq_length, 0);
  std::vector<int32_t> width_position_ids(input_seq_length, 0);

  int pos_idx = 0;
  size_t img_idx = 0;
  for (size_t i = 0; i < seq_length;) {
    if (input_ids[i] == IMAGE_TOKEN_ID && img_idx < image_grid_thw.size()) {
      auto [grid_t, grid_h, grid_w] = image_grid_thw[img_idx];
      int num_image_tokens = grid_t * grid_h * grid_w;

      for (int t = 0; t < grid_t; t++) {
        for (int h = 0; h < grid_h; h++) {
          for (int w = 0; w < grid_w; w++) {
            size_t token_offset = i + (t * grid_h * grid_w) + (h * grid_w) + w;
            if (token_offset < input_ids.size()) {
              time_position_ids[token_offset] = pos_idx + t;
              height_position_ids[token_offset] = pos_idx + h;
              width_position_ids[token_offset] = pos_idx + w;
            }
          }
        }
      }
      pos_idx += std::max({grid_t, grid_h, grid_w});
      i += num_image_tokens;
      img_idx++;
    } else {
      time_position_ids[i] = pos_idx;
      height_position_ids[i] = pos_idx;
      width_position_ids[i] = pos_idx;
      pos_idx++;
      i++;
    }
  }

  for (size_t i = seq_length; i < static_cast<size_t>(input_seq_length); i++) {
    time_position_ids[i] = time_position_ids[i - 1] + 1;
    height_position_ids[i] = height_position_ids[i - 1] + 1;
    width_position_ids[i] = width_position_ids[i - 1] + 1;
  }

  // Prepare deepstack embeddings (zeros for now, should be properly mapped)
  std::vector<half_float::half> deepstack_embed_0(
      input_seq_length * embedding_length_, half_float::half(0.0f));
  std::vector<half_float::half> deepstack_embed_1(
      input_seq_length * embedding_length_, half_float::half(0.0f));
  std::vector<half_float::half> deepstack_embed_2(
      input_seq_length * embedding_length_, half_float::half(0.0f));

  // Scatter deepstack features (similar to image features)
  size_t ds_feature_idx = 0;
  for (size_t i = 0;
       i < input_ids.size() && ds_feature_idx < deepstack_0.size(); i++) {
    if (input_ids[i] == IMAGE_TOKEN_ID) {
      size_t num_image_tokens = 0;
      for (size_t j = i; j < input_ids.size() && input_ids[j] == IMAGE_TOKEN_ID;
           j++) {
        num_image_tokens++;
      }

      for (size_t j = 0;
           j < num_image_tokens && ds_feature_idx < deepstack_0.size(); j++) {
        size_t embed_offset = (i + j) * embedding_length_;
        size_t feat_size = embedding_length_;
        if (ds_feature_idx + feat_size <= deepstack_0.size()) {
          std::memcpy(deepstack_embed_0.data() + embed_offset,
                      deepstack_0.data() + ds_feature_idx,
                      feat_size * sizeof(half_float::half));
          std::memcpy(deepstack_embed_1.data() + embed_offset,
                      deepstack_1.data() + ds_feature_idx,
                      feat_size * sizeof(half_float::half));
          std::memcpy(deepstack_embed_2.data() + embed_offset,
                      deepstack_2.data() + ds_feature_idx,
                      feat_size * sizeof(half_float::half));
        }
        ds_feature_idx += feat_size;
      }
      i += num_image_tokens - 1;
    }
  }

  // Run prefill
  Timer prefill_timer;
  prefill_timer.start();

  int32_t valid_length = 0;
  int32_t current_length = 0;
  past_seq_len_ = 0;

  int prefill_loop_round = (seq_length + prefill_length_ - 1) / prefill_length_;

  for (int round = 0; round < prefill_loop_round; round++) {
    int start = round * prefill_length_;
    int end =
        std::min((round + 1) * prefill_length_, static_cast<int>(seq_length));
    current_length = end - start;

    // Build fixed-size chunk tensors to match prefill input shape.
    // The tail chunk may be shorter than prefill_length_, so we zero-pad it.
    std::vector<half_float::half> chunk_embeds(
        prefill_length_ * embedding_length_, half_float::half(0.0f));
    std::vector<int32_t> chunk_time(prefill_length_, 0);
    std::vector<int32_t> chunk_height(prefill_length_, 0);
    std::vector<int32_t> chunk_width(prefill_length_, 0);
    std::vector<half_float::half> chunk_ds0(prefill_length_ * embedding_length_,
                                            half_float::half(0.0f));
    std::vector<half_float::half> chunk_ds1(prefill_length_ * embedding_length_,
                                            half_float::half(0.0f));
    std::vector<half_float::half> chunk_ds2(prefill_length_ * embedding_length_,
                                            half_float::half(0.0f));

    size_t token_span = static_cast<size_t>(current_length);
    size_t embed_span = token_span * static_cast<size_t>(embedding_length_);
    size_t embed_start = static_cast<size_t>(start) * embedding_length_;

    if (embed_span > 0) {
      std::memcpy(chunk_embeds.data(), inputs_embeds.data() + embed_start,
                  embed_span * sizeof(half_float::half));
      std::memcpy(chunk_ds0.data(), deepstack_embed_0.data() + embed_start,
                  embed_span * sizeof(half_float::half));
      std::memcpy(chunk_ds1.data(), deepstack_embed_1.data() + embed_start,
                  embed_span * sizeof(half_float::half));
      std::memcpy(chunk_ds2.data(), deepstack_embed_2.data() + embed_start,
                  embed_span * sizeof(half_float::half));
      std::memcpy(chunk_time.data(), time_position_ids.data() + start,
                  token_span * sizeof(int32_t));
      std::memcpy(chunk_height.data(), height_position_ids.data() + start,
                  token_span * sizeof(int32_t));
      std::memcpy(chunk_width.data(), width_position_ids.data() + start,
                  token_span * sizeof(int32_t));
    }

    Timer prefill_set_input_timer;
    prefill_set_input_timer.start();
    PrefillSetInputDatas(chunk_embeds, chunk_time, chunk_height, chunk_width,
                         valid_length, current_length, chunk_ds0, chunk_ds1,
                         chunk_ds2);
    prefill_set_input_timer.end();
    perf_info_.prefill_set_input_time += prefill_set_input_timer.elapsed_ms();

    Timer prefill_infer_timer;
    prefill_infer_timer.start();
    PrefillInfer();
    prefill_infer_timer.end();
    perf_info_.prefill_infer_time += prefill_infer_timer.elapsed_ms();

    valid_length += current_length;
    past_seq_len_ = valid_length;
  }

  prefill_timer.end();
  perf_info_.prefill_time = prefill_timer.elapsed_ms();

  // Get first token
  Timer prefill_get_output_timer;
  prefill_get_output_timer.start();
  int next_token = PrefillGetOutputDatas();
  prefill_get_output_timer.end();
  perf_info_.prefill_get_output_time = prefill_get_output_timer.elapsed_ms();
  generated_ids_.clear();
  generated_ids_.push_back(next_token);

  perf_info_.ttft_time = perf_info_.prefill_time + perf_info_.vision_time;

  // Decode phase
  std::vector<int> chat_history_ids(input_ids.begin(),
                                    input_ids.begin() + seq_length);

  std::vector<int> prefill_ids = {next_token};
  std::string prefill_response = tokenizer_->Decode(prefill_ids);

  chat_history_ids.push_back(next_token);

  std::string all_response = prefill_response;
  int context_length = seq_length;
  slide_len_ = 10;
  skip_tokens_ = 0;

  std::vector<int> slide_window_ids(
      chat_history_ids.end() -
          std::min(slide_len_, (int)chat_history_ids.size()),
      chat_history_ids.end());
  last_response_ = tokenizer_->Decode(slide_window_ids);
  std::string decode_response;

  std::cout << "Response: " << prefill_response << std::flush;
  Timer decode_timer;
  decode_timer.start();

  // Save rope_deltas for decode
  rope_deltas_ = pos_idx - seq_length;

  while (true) {
    if (context_length >= context_max_length_ || next_token == EOS_TOKEN_ID) {
      std::cout << decode_response << std::endl;
      all_response += decode_response;
      break;
    }

    // Get embedding for single token
    std::vector<int> single_token = {next_token};
    Timer decode_embedding_timer;
    decode_embedding_timer.start();
    auto token_embeds = tokenizer_->EmbeddingTokens(single_token);
    decode_embedding_timer.end();
    perf_info_.decode_embedding_time += decode_embedding_timer.elapsed_ms();
    std::vector<half_float::half> decode_embeds(
        token_embeds, token_embeds + embedding_length_);

    // Calculate position IDs for decode
    int32_t delta = past_seq_len_ + rope_deltas_;
    std::vector<int32_t> decode_time = {delta};
    std::vector<int32_t> decode_height = {delta};
    std::vector<int32_t> decode_width = {delta};
    // printf("Decode position IDs - time: %d, height: %d, width: %d\n", delta,
    //       delta, delta);

    // Deepstack embeddings for decode (zeros)
    std::vector<half_float::half> decode_ds0(embedding_length_,
                                             half_float::half(0));
    std::vector<half_float::half> decode_ds1(embedding_length_,
                                             half_float::half(0));
    std::vector<half_float::half> decode_ds2(embedding_length_,
                                             half_float::half(0));

    Timer decode_set_input_timer;
    decode_set_input_timer.start();
    DecodeSetInputDatas(decode_embeds, decode_time, decode_height, decode_width,
                        context_length, decode_ds0, decode_ds1, decode_ds2);
    decode_set_input_timer.end();
    perf_info_.decode_set_input_time += decode_set_input_timer.elapsed_ms();

    Timer decode_infer_timer;
    decode_infer_timer.start();
    DecodeInfer();
    decode_infer_timer.end();
    perf_info_.decode_infer_time += decode_infer_timer.elapsed_ms();

    Timer decode_get_output_timer;
    decode_get_output_timer.start();
    next_token = DecodeGetOutputDatas();
    decode_get_output_timer.end();
    perf_info_.decode_get_output_time += decode_get_output_timer.elapsed_ms();
    generated_ids_.push_back(next_token);
    chat_history_ids.push_back(next_token);
    context_length++;
    past_seq_len_++;
    if (next_token == EOS_TOKEN_ID) {
      std::cout << decode_response << std::endl;
      all_response += decode_response;
      break;
    }

    int substart = utf8_len(last_response_);
    std::vector<int> decode_window_ids(
        chat_history_ids.end() - slide_len_ - skip_tokens_ - 1,
        chat_history_ids.end());
    Timer decode_tokenize_timer;
    decode_tokenize_timer.start();
    std::string tmp_response = tokenizer_->Decode(decode_window_ids);
    decode_tokenize_timer.end();
    perf_info_.decode_tokenization_time += decode_tokenize_timer.elapsed_ms();
    std::u32string udecode_response =
        utf8_to_u32(tmp_response).substr(substart);
    decode_response = u32_to_utf8(udecode_response);

    if (decode_response != "" && IsValidChar(udecode_response.back())) {
      std::cout << decode_response << std::flush;
      all_response += decode_response;
      std::vector<int> cur_slide_win(
          chat_history_ids.end() -
              std::min(slide_len_, (int)chat_history_ids.size()),
          chat_history_ids.end());
      last_response_ = tokenizer_->Decode(cur_slide_win);
      skip_tokens_ = 0;
    } else {
      skip_tokens_ += 1;
    }
  }

  decode_timer.end();
  perf_info_.decode_time = decode_timer.elapsed_ms();
  timer.end();
  perf_info_.total_time = timer.elapsed_ms();
  perf_info_.output_tokens = generated_ids_.size();
  perf_info_.embedding_time =
      perf_info_.prefill_embedding_time + perf_info_.decode_embedding_time;

  std::cout << std::endl;

  // Print performance metrics in Python demo summary style.
  auto safe_speed = [](float cnt, float ms) {
    return ms > 0.0f ? cnt / (ms * 0.001f) : 0.0f;
  };
  auto safe_ms_per_token = [](float ms, int tokens) {
    return tokens > 0 ? ms / static_cast<float>(tokens) : 0.0f;
  };
  const int decode_api_tokens =
      perf_info_.output_tokens > 0 ? perf_info_.output_tokens - 1 : 0;
  float tpot = perf_info_.output_tokens > 0
                   ? perf_info_.decode_time / perf_info_.output_tokens
                   : 0.0f;

  std::cout << std::fixed << std::setprecision(3);
  std::cout << "[SUCCESS] "
               "==============================================================="
               "====================================="
            << std::endl;
  std::cout << "[SUCCESS]                     Model Inference Performance "
               "Summary Report"
            << std::endl;
  std::cout << "[SUCCESS] "
               "==============================================================="
               "====================================="
            << std::endl;

  std::cout << "[SUCCESS] Configuration Details:" << std::endl;
  std::cout << "[SUCCESS]   Batch Size:      " << perf_info_.batch_size
            << std::endl;
  std::cout << "[SUCCESS]   Input Length per Sample:    "
            << perf_info_.input_tokens << " tokens" << std::endl;
  std::cout << "[SUCCESS]   Output Length per Sample:     "
            << perf_info_.output_tokens << " tokens" << std::endl;
  std::cout << "[SUCCESS]   Number of Images:      " << perf_info_.num_images
            << " images" << std::endl;
  std::cout << std::setprecision(2) << "[SUCCESS]   Prefill Model Load Time: "
            << perf_info_.prefill_model_load_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   Decode Model Load Time:  "
            << perf_info_.decode_model_load_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   Vision Model Load Time:  "
            << perf_info_.vision_model_load_time << "ms" << std::endl;

  std::cout << "[SUCCESS] Vision Stage Performance:" << std::endl;
  std::cout << "[SUCCESS]   Total Time:  " << perf_info_.vision_time
            << "ms | Speed:   "
            << safe_speed(static_cast<float>(perf_info_.num_images),
                          perf_info_.vision_time)
            << " images/s" << std::endl;
  std::cout << "[SUCCESS]   Preprocessing Time: "
            << perf_info_.vision_preprocess_time << "ms | Speed:  "
            << safe_speed(static_cast<float>(perf_info_.num_images),
                          perf_info_.vision_preprocess_time)
            << " images/s" << std::endl;
  std::cout << "[SUCCESS]   API SetInput Time:   "
            << perf_info_.vision_set_input_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   API Inference Time: "
            << perf_info_.vision_infer_time << "ms | Speed:  "
            << safe_speed(static_cast<float>(perf_info_.num_images),
                          perf_info_.vision_infer_time)
            << " images/s" << std::endl;
  std::cout << "[SUCCESS]   API GetOutput Time:  "
            << perf_info_.vision_get_output_time << "ms" << std::endl;

  std::cout << "[SUCCESS] Prefill Stage Performance:" << std::endl;
  std::cout << "[SUCCESS]   Total Time:  " << perf_info_.prefill_time
            << "ms | Speed: "
            << safe_speed(static_cast<float>(perf_info_.input_tokens),
                          perf_info_.prefill_time)
            << " tokens/s" << std::endl;
  std::cout << "[SUCCESS]   Tokenization Time:   "
            << perf_info_.prefill_tokenization_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   Embedding Time:    "
            << perf_info_.prefill_embedding_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   API SetInput Time:   "
            << perf_info_.prefill_set_input_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   API Inference Time: "
            << perf_info_.prefill_infer_time << "ms | Prefill Speed: "
            << safe_speed(static_cast<float>(perf_info_.input_tokens),
                          perf_info_.prefill_infer_time)
            << " tokens/s" << std::endl;
  std::cout << "[SUCCESS]   API GetOutput Time:  "
            << perf_info_.prefill_get_output_time << "ms" << std::endl;

  std::cout << "[SUCCESS] Decode Stage Performance:" << std::endl;
  std::cout << "[SUCCESS]   Total Time: " << perf_info_.decode_time
            << "ms | Speed:   "
            << safe_speed(static_cast<float>(perf_info_.output_tokens),
                          perf_info_.decode_time)
            << " tokens/s" << std::endl;
  std::cout << "[SUCCESS]   Tokenization Time:    "
            << perf_info_.decode_tokenization_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   Embedding Time:    "
            << perf_info_.decode_embedding_time << "ms" << std::endl;
  std::cout << "[SUCCESS]   API SetInput Time:   "
            << safe_ms_per_token(perf_info_.decode_set_input_time,
                                 decode_api_tokens)
            << "ms/token" << std::endl;
  std::cout << "[SUCCESS]   API Inference Time: "
            << safe_ms_per_token(perf_info_.decode_infer_time,
                                 decode_api_tokens)
            << "ms/token | Decode Speed:   "
            << safe_speed(static_cast<float>(perf_info_.output_tokens),
                          perf_info_.decode_infer_time)
            << " tokens/s" << std::endl;
  std::cout << "[SUCCESS]   API GetOutput Time:  "
            << safe_ms_per_token(perf_info_.decode_get_output_time,
                                 decode_api_tokens)
            << "ms/token" << std::endl;

  std::cout << std::setprecision(3)
            << "[SUCCESS] Overall Performance Metrics:" << std::endl;
  std::cout << "[SUCCESS]   TTFT (Time To First Token):  "
            << perf_info_.ttft_time << " ms" << std::endl;
  std::cout << "[SUCCESS]   TPOT (Time Per Output Token): " << tpot
            << " ms/token" << std::endl;
  std::cout << "[SUCCESS]   E2E Latency (End-to-End):      "
            << perf_info_.total_time * 0.001f << " seconds" << std::endl;
  std::cout << std::setprecision(2)
            << "[SUCCESS]   E2E TPS (Throughput):         "
            << safe_speed(static_cast<float>(perf_info_.output_tokens),
                          perf_info_.total_time)
            << " tokens/s" << std::endl;
  std::cout << "[SUCCESS] "
               "==============================================================="
               "====================================="
            << std::endl;

  return all_response;
}