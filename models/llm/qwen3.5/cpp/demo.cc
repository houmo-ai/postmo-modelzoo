/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Qwen3.5 MLLM inference demo using Houmo framework.
 *
 *   Feature showcase:
 *     1. Basic inference - StreamingDecoder streaming output
 *     2. Image understanding - Qwen35MLLMContext image processing
 *     3. Advanced parameters - max tokens and model info
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

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <regex>
#include <string>
#include <unordered_map>
#include <vector>

#include "core/model_factory.h"
#include "modules/streaming_decoder.h"
#include "qwen35_mllm_model.h"

namespace fs = std::filesystem;

// ============================================================================
// Helper functions
// ============================================================================

static void printUsage(const char* program_name) {
  std::cout << "Qwen3.5 MLLM C++ Inference Demo\n\n";
  std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
  std::cout << "Options:\n";
  std::cout << "  --config <path>       Path to config.yaml\n";
  std::cout << "  --model_name <name>   Model name, e.g. qwen3.5\n";
  std::cout << "  --model_size <size>   Model size, e.g. 9b\n";
  std::cout
      << "  --prefill <path>      Path to prefill model (.hmm) [required]\n";
  std::cout
      << "  --decode <path>       Path to decode model (.hmm) [required]\n";
  std::cout << "  --visual <path>       Path to visual model (.hmm), required "
               "for image input\n";
  std::cout
      << "  --embedding <path>    Path to embedding file (.bin) [required]\n";
  std::cout << "  --tokenizer <path>    Path to tokenizer json or tokenizer "
               "directory [required]\n";
  std::cout << "  --prompt <text>       Prompt text, default: 描述这些图片\n";
  std::cout << "  --system_prompt <text> System prompt override\n";
  std::cout << "  --image <path>        Image path, can be specified multiple "
               "times\n";
  std::cout << "  --image_path <path>   Alias for --image\n";
  std::cout
      << "  --max-tokens <num>    Maximum generated tokens, default 256\n";
  std::cout << "  --device <id>         Device id, default 0\n";
  std::cout << "  --ndevice <num>       Number of devices, only 1 is supported\n";
  std::cout << "  --batch <num>         Batch size, only 1 is supported\n";
  std::cout << "  -h, --help            Show this help message\n\n";
  std::cout << "Examples:\n";
  std::cout << "  # Text inference\n";
  std::cout << "  " << program_name
            << " --prefill model_prefill.hmm --decode model_decode.hmm "
               "--embedding quant_embedding.bin --tokenizer tokenizer.json "
               "--prompt \"Introduce yourself\"\n\n";
  std::cout << "  # Image understanding\n";
  std::cout << "  " << program_name
            << " --prefill model_prefill.hmm --decode model_decode.hmm "
               "--visual model_visual.hmm --embedding quant_embedding.bin "
               "--tokenizer tokenizer.json --image test.jpg "
               "--prompt \"Describe this image\"\n";
}

static void printSeparator(const char* title) {
  std::cout << "\n========== " << title << " ==========\n";
}

static bool requireFile(const std::string& flag, const std::string& path) {
  if (path.empty() || !fs::exists(path)) {
    std::cerr << "Error: " << flag << " path does not exist: " << path << "\n";
    return false;
  }
  return true;
}

struct ModelSelection {
  std::string name;
  std::string size;
  std::string repo_name;
  int ndevice = 1;
};

static std::string trim(std::string value) {
  const auto first = value.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return {};
  return value.substr(first, value.find_last_not_of(" \t\r\n") - first + 1);
}

static std::string yaml_value(const std::string& line) {
  const auto colon = line.find(':');
  if (colon == std::string::npos) return {};
  std::string value = trim(line.substr(colon + 1));
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    value = value.substr(1, value.size() - 2);
  }
  return value;
}

static std::string unquote(std::string value) {
  value = trim(std::move(value));
  if (value.size() >= 2 && value.front() == '"' && value.back() == '"') {
    return value.substr(1, value.size() - 2);
  }
  return value;
}

static void read_model_config_line(
    const std::string& line, std::string& default_name, std::string& default_size,
    std::string& current_name, std::string& current_size,
    std::unordered_map<std::string, std::string>& values) {
  const std::string stripped = trim(line);
  if (stripped.empty() || stripped[0] == '#') return;
  const size_t indent = line.find_first_not_of(' ');
  if (indent == 0 && stripped.rfind("default_model_name:", 0) == 0) {
    default_name = yaml_value(stripped);
  } else if (indent == 0 && stripped.rfind("default_model_size:", 0) == 0) {
    default_size = yaml_value(stripped);
  } else if (indent == 2 && stripped.back() == ':') {
    current_name = unquote(stripped.substr(0, stripped.size() - 1));
  } else if (indent == 4 && stripped.back() == ':') {
    current_size = unquote(stripped.substr(0, stripped.size() - 1));
  } else if (indent >= 6 && !current_name.empty() && !current_size.empty()) {
    values[current_name + "\n" + current_size + "\n" + stripped.substr(0, stripped.find(':'))] = yaml_value(stripped);
  }
}

static ModelSelection load_model_selection(
    const std::string& path, std::string name, std::string size) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("failed to open config: " + path);
  std::string default_name, default_size, current_name, current_size, line;
  std::unordered_map<std::string, std::string> values;
  while (std::getline(input, line)) {
    read_model_config_line(line, default_name, default_size, current_name, current_size, values);
  }
  if (name.empty()) name = default_name;
  if (size.empty()) size = default_size;
  ModelSelection result{name, size, name + "-" + size, 1};
  const std::string key = name + "\n" + size + "\n";
  if (values.count(key + "modelscope_repo")) {
    std::string repo = values[key + "modelscope_repo"];
    const auto quote = repo.find('"');
    if (quote != std::string::npos) repo = repo.substr(quote + 1);
    const auto slash = repo.find_last_of('/');
    if (slash != std::string::npos) repo = repo.substr(slash + 1);
    const auto end = repo.find_first_of("\"'] ,");
    result.repo_name = repo.substr(0, end);
  }
  if (values.count(key + "ndevice")) result.ndevice = std::stoi(values[key + "ndevice"]);
  return result;
}

static std::string env_or(const char* key, const std::string& fallback) {
  const char* value = std::getenv(key);
  return value && *value ? value : fallback;
}

static std::string find_default_visual_path(
    const fs::path& output_dir, const std::string& model_prefix) {
  const fs::path visual_fallback = output_dir / (model_prefix + "_visual.hmm");
  for (const auto& entry : fs::directory_iterator(output_dir)) {
    if (!entry.is_regular_file()) continue;
    const std::string name = entry.path().filename().string();
    if (name.rfind(model_prefix + "_visual_m", 0) == 0 &&
        entry.path().extension() == ".hmm") {
      return output_dir.string();
    }
  }
  return visual_fallback.string();
}

struct Message {
  std::string role;
  std::string content;
};

static std::string ApplyChatTemplate(const std::vector<Message>& msgs,
                                     bool add_generation_prompt,
                                     bool enable_thinking) {
  std::string out;
  out.reserve(1024);

  for (const auto& m : msgs) {
    out.append("<|im_start|>");
    out.append(m.role);
    out.push_back('\n');
    out.append(m.content);
    out.append("<|im_end|>\n");
  }

  if (add_generation_prompt) {
    out.append("<|im_start|>assistant\n");
  }

  if (!enable_thinking) {
    out.append("<think>\n\n</think>\n\n");
  }

  return out;
}

static void printModelInfo(houmo::LLMModel& model) {
  printSeparator("Model Information");

  auto info = model.model_info();
  std::cout << "Model Name:       " << info.model_name << "\n";
  std::cout << "Model Type:       "
            << (info.type == houmo::ModelType::VLM ? "VLM" : "LLM") << "\n";
  std::cout << "n_batch:          " << info.n_batch << "\n";
  std::cout << "n_vocab:          " << info.n_vocab << "\n";
  std::cout << "n_embd:           " << info.n_embd << "\n";
  std::cout << "n_layer:          " << info.n_layer << "\n";
  std::cout << "n_ctx:            " << info.n_ctx << "\n";
  std::cout << "prefill_length:   " << info.prefill_length << "\n";
  std::cout << "kv_cache_layers:  " << info.kv_cache_layers << "\n";
  std::cout << "max_ctx_available:" << model.max_ctx_available() << "\n";
  std::cout << "vocab_size:       " << model.vocab_size() << "\n";
  std::cout << "embedding_dim:    " << model.embedding_dim() << "\n";
}

static std::string BuildImagePromptPrefix(size_t image_count) {
  std::string content;
  for (size_t i = 0; i < image_count; ++i) {
    content += "<|vision_start|><|image_pad|><|vision_end|>";
  }
  return content;
}

static void printGenerationMetrics(houmo::Context& ctx) {
  std::cout << "\n";
  ctx.profiler().print_summary();

  const auto stats = ctx.perf_stats();
  std::cout << "\n=== Generation Metrics ===\n"
            << "Input Tokens:         " << stats.n_input_tokens << "\n"
            << "Output Tokens:        " << stats.n_output_tokens << "\n"
            << "Prefill Time:         " << stats.prefill_time_ms << " ms\n"
            << "Decode Time:          " << stats.decode_time_ms << " ms\n"
            << "Total Time:           " << stats.total_time_ms << " ms\n"
            << "TTFT:                 " << stats.ttft_ms << " ms\n"
            << "TPS:                  " << stats.tps << " tok/s\n";
}

// ============================================================================
// Example features
// ============================================================================

static void exampleBasicInference(houmo::LLMModel& model,
                                  const std::string& prompt,
                                  const std::string& system_prompt,
                                  const houmo::SamplingParams& params,
                                  bool enable_thinking) {
  printSeparator("Basic Inference");

  auto ctx = model.create_context();
  // Match Qwen35Engine.generate(): text-only requests use this default
  // system prompt when the caller does not provide one.
  std::vector<Message> msgs = {
      {"system", system_prompt.empty() ? "You are a helpful assistant." : system_prompt},
      {"user", prompt},
  };
  std::string rendered = ApplyChatTemplate(msgs, true, enable_thinking);
  auto tokens = model.tokenize(rendered, false, false);

  std::cout << "Prompt: " << prompt << "\n";
  std::cout << "Output: ";

  houmo::StreamingDecoder decoder(model.tokenizer());
  ctx->generate(tokens, params, [&decoder](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";

  printGenerationMetrics(*ctx);
}

static void exampleImageUnderstanding(houmo::LLMModel& model,
                                      const std::vector<std::string>& images,
                                      const std::string& prompt,
                                      const std::string& system_prompt,
                                      const houmo::SamplingParams& params,
                                      bool enable_thinking) {
  printSeparator("Image Understanding");

  auto ctx = model.create_context();
  auto* qwen_ctx = dynamic_cast<houmo::Qwen35MLLMContext*>(ctx.get());
  if (!qwen_ctx) {
    std::cerr << "Error: failed to create Qwen35MLLMContext\n";
    return;
  }

  qwen_ctx->set_images(images);
  std::cout << "Loaded " << images.size() << " images (Qwen35MLLM)\n";

  std::string content = BuildImagePromptPrefix(images.size()) + prompt;
  // Match Qwen35Engine.generate(): multimodal requests use this default
  // system prompt when the caller does not provide one.
  std::vector<Message> msgs = {
      {"system", system_prompt.empty() ? "介绍一下这些图片" : system_prompt},
      {"user", content},
  };
  std::string rendered = ApplyChatTemplate(msgs, true, enable_thinking);
  auto tokens = model.tokenize(rendered, false, false);

  std::cout << "Prompt: " << prompt << "\n";
  std::cout << "Output: ";

  houmo::StreamingDecoder decoder(model.tokenizer());
  qwen_ctx->generate(tokens, params, [&decoder](houmo::Token token) {
    std::string chunk = decoder.decode(token);
    if (!chunk.empty()) {
      std::cout << chunk << std::flush;
    }
    return true;
  });
  std::cout << "\n";

  printGenerationMetrics(*qwen_ctx);
}

// ============================================================================
// Main function
// ============================================================================

int main(int argc, char* argv[]) {
  std::string config_path = "config.yaml";
  std::string model_name;
  std::string model_size;
  std::string prefill_path;
  std::string decode_path;
  std::string visual_path;
  std::string embedding_path;
  std::string tokenizer_path;
  std::string prompt = "描述这些图片";
  std::string system_prompt;
  std::vector<std::string> image_paths;
  int max_tokens = 0;
  int device_id = 0;
  int ndevice = 1;
  int batch_size = 1;
  float temperature = 1.0f;
  float top_p = 1.0f;
  int top_k = 1;
  float presence_penalty = 0.0f;
  float repetition_penalty = 1.0f;

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    if (arg == "--help" || arg == "-h") {
      printUsage(argv[0]);
      return 0;
    } else if (arg == "--config" && i + 1 < argc) {
      config_path = argv[++i];
    } else if (arg == "--model_name" && i + 1 < argc) {
      model_name = argv[++i];
    } else if (arg == "--model_size" && i + 1 < argc) {
      model_size = argv[++i];
    } else if ((arg == "--prefill" || arg == "--prefill_path") && i + 1 < argc) {
      prefill_path = argv[++i];
    } else if ((arg == "--decode" || arg == "--decode_path") && i + 1 < argc) {
      decode_path = argv[++i];
    } else if ((arg == "--visual" || arg == "--vision_path") && i + 1 < argc) {
      visual_path = argv[++i];
    } else if ((arg == "--embedding" || arg == "--embedding_path") && i + 1 < argc) {
      embedding_path = argv[++i];
    } else if ((arg == "--tokenizer" || arg == "--tokenizer_dir") && i + 1 < argc) {
      tokenizer_path = argv[++i];
    } else if ((arg == "--prompt" || arg == "--question") && i + 1 < argc) {
      prompt = argv[++i];
    } else if (arg == "--system_prompt" && i + 1 < argc) {
      system_prompt = argv[++i];
    } else if ((arg == "--image" || arg == "--image_path") && i + 1 < argc) {
      image_paths.emplace_back(argv[++i]);
    } else if ((arg == "--max-tokens" || arg == "--max-new-tokens") && i + 1 < argc) {
      max_tokens = std::stoi(argv[++i]);
    } else if (arg == "--device" && i + 1 < argc) {
      device_id = std::stoi(argv[++i]);
    } else if (arg == "--ndevice" && i + 1 < argc) {
      ndevice = std::stoi(argv[++i]);
    } else if (arg == "--batch" && i + 1 < argc) {
      batch_size = std::stoi(argv[++i]);
    } else if (arg == "--temperature" && i + 1 < argc) {
      temperature = std::stof(argv[++i]);
    } else if ((arg == "--topk" || arg == "--top-k") && i + 1 < argc) {
      top_k = std::stoi(argv[++i]);
    } else if ((arg == "--topp" || arg == "--top-p") && i + 1 < argc) {
      top_p = std::stof(argv[++i]);
    } else if (arg == "--presence-penalty" && i + 1 < argc) {
      presence_penalty = std::stof(argv[++i]);
    } else if (arg == "--repetition-penalty" && i + 1 < argc) {
      repetition_penalty = std::stof(argv[++i]);
    } else {
      std::cerr << "Unknown or incomplete option: " << arg << "\n\n";
      printUsage(argv[0]);
      return 1;
    }
  }

  fs::path config_file = fs::absolute(config_path);
  if (!fs::exists(config_file) && config_path == "config.yaml") {
    const fs::path parent_config = fs::current_path().parent_path() / "config.yaml";
    if (fs::exists(parent_config)) config_file = parent_config;
  }
  if (batch_size != 1) {
    std::cerr << "Error: Qwen3.5 C++ demo only supports --batch 1\n";
    return 1;
  }
  if (ndevice != 1) {
    std::cerr << "Error: Qwen3.5 C++ demo only supports --ndevice 1\n";
    return 1;
  }
  const ModelSelection selected = load_model_selection(
      config_file.string(), model_name, model_size);
  const fs::path model_root = config_file.parent_path();
  const fs::path output_dir = model_root / "output" / env_or("HOUMO_TARGET", "xh2");
  const std::string model_prefix = selected.name + "-" + selected.size;
  if (prefill_path.empty()) prefill_path = (output_dir / (model_prefix + "_prefill.hmm")).string();
  if (decode_path.empty()) decode_path = (output_dir / (model_prefix + "_decode.hmm")).string();
  if (embedding_path.empty()) embedding_path = (output_dir / "hmquant/quant_embedding.bin").string();
  if (tokenizer_path.empty()) tokenizer_path = (model_root / selected.repo_name).string();
  if (visual_path.empty()) visual_path = find_default_visual_path(output_dir, model_prefix);
  if (image_paths.empty()) {
    image_paths.push_back((fs::path(env_or("HOUMO_EXAMPLES_PATH", model_root.string())) / "data/pic/beach.jpeg").string());
  }

  if (!requireFile("--prefill", prefill_path) ||
      !requireFile("--decode", decode_path) ||
      !requireFile("--embedding", embedding_path) ||
      !requireFile("--tokenizer", tokenizer_path)) {
    printUsage(argv[0]);
    return 1;
  }

  for (const auto& image_path : image_paths) {
    if (!requireFile("--image", image_path)) {
      return 1;
    }
  }

  if (!image_paths.empty() && !requireFile("--visual", visual_path)) {
    printUsage(argv[0]);
    return 1;
  }

  houmo::ModelConfig config;
  config.devices = {device_id};
  config.prefill_path = prefill_path;
  config.decode_path = decode_path;
  config.vision_path = visual_path;
  config.embedding_path = embedding_path;
  config.tokenizer_path = tokenizer_path;

  try {
    std::cout << "========================================\n";
    std::cout << "  Qwen3.5 MLLM C++ Inference Demo\n";
    std::cout << "  Houmo Inference Framework v" << houmo::version() << "\n";
    std::cout << "========================================\n\n";

    const auto model_series = houmo::ModelSeries::kQwen35MLLM;
    std::cout << "Loading registered model: "
              << houmo::ModelSeriesToString(model_series) << "...\n";
    auto model =
        houmo::ModelFactory<houmo::LLMModel>::Create(model_series, config);
    if (!model) {
      std::cerr << "Error: model is not registered: "
                << houmo::ModelSeriesToString(model_series) << "\n";
      return 2;
    }
    std::cout << "Model loaded successfully!\n";

    printModelInfo(*model);

    houmo::SamplingParams params;
    params.max_tokens = max_tokens;
    params.temperature = temperature;
    params.top_p = top_p;
    params.top_k = top_k;
    params.presence_penalty = presence_penalty;
    params.repetition_penalty = repetition_penalty;

    bool enable_thinking = false;
    if (!image_paths.empty()) {
      exampleImageUnderstanding(*model, image_paths, prompt, system_prompt,
                                params, enable_thinking);
    } else {
      exampleBasicInference(*model, prompt, system_prompt, params, enable_thinking);
    }

    std::cout << "\nDone!\n";
  } catch (const houmo::Exception& e) {
    std::cerr << "Houmo Error: " << e.what() << "\n";
    return 3;
  } catch (const std::exception& e) {
    std::cerr << "Standard Error: " << e.what() << "\n";
    return 3;
  }

  return 0;
}
