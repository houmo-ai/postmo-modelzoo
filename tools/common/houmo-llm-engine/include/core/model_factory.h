/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: model_factory.h
 * Description:
 *   Template-based model factory supporting static registration and runtime
 *   model type selection. Each model type (LLM, ASR, TTS) has its own
 *   factory instance with unified interface.
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

#pragma once

#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include "base/houmo.h"

namespace houmo {

// ============================================================================
// Model Series Enumeration
// ============================================================================

/**
 * @brief Model series enumeration
 */
enum class ModelSeries {
  kUnknown,      // Unknown type
  kQwen3LLM,     // Qwen3 pure-text LLM
  kQwen35MLLM,   // Qwen3.5 multimodal MLLM
  kQwen3VLM,     // Qwen3-VL vision-language
  kWhisperASR,   // Whisper ASR
  kGlmAsr,       // GLM-ASR
  kQwen3Asr,     // Qwen3-ASR
};

/**
 * @brief Convert model series to string
 */
inline std::string ModelSeriesToString(ModelSeries series) {
  switch (series) {
    case ModelSeries::kQwen3LLM:
      return "qwen3_llm";
    case ModelSeries::kQwen35MLLM:
      return "qwen35_mllm";
    case ModelSeries::kQwen3VLM:
      return "qwen3_vlm";
    case ModelSeries::kWhisperASR:
      return "whisper_asr";
    case ModelSeries::kGlmAsr:
      return "glm_asr";
    case ModelSeries::kQwen3Asr:
      return "qwen3_asr";
    default:
      return "unknown";
  }
}

/**
 * @brief Convert string to model series
 */
inline ModelSeries StringToModelSeries(const std::string& str) {
  if (str == "qwen3_llm" || str == "qwen3") {
    return ModelSeries::kQwen3LLM;
  } else if (str == "qwen35_mllm" || str == "qwen35") {
    return ModelSeries::kQwen35MLLM;
  } else if (str == "qwen3_vlm") {
    return ModelSeries::kQwen3VLM;
  } else if (str == "whisper_asr" || str == "whisper") {
    return ModelSeries::kWhisperASR;
  } else if (str == "glm_asr" || str == "glm-asr") {
    return ModelSeries::kGlmAsr;
  } else if (str == "qwen3_asr" || str == "qwen3-asr") {
    return ModelSeries::kQwen3Asr;
  }
  return ModelSeries::kUnknown;
}

// ============================================================================
// Template-based Model Factory
// ============================================================================

/**
 * @brief Template-based model factory
 *
 * Each model type (LLMModel, ASRModel, TTSModel) has its own factory instance.
 * This provides type-safe model creation with unified interface.
 *
 * Usage:
 * ```cpp
 * // Create models
 * auto llm = ModelFactory<LLMModel>::Create(ModelSeries::kQwen3LLM, config);
 * auto asr = ModelFactory<ASRModel>::Create(ModelSeries::kWhisperASR, config);
 *
 * // Register models
 * ModelFactory<LLMModel>::Register("qwen3_llm", ModelSeries::kQwen3LLM,
 *     [](const ModelConfig& c) { return std::make_unique<Qwen3LLMModel>(c); },
 *     "Qwen3 LLM");
 * ```
 *
 * @tparam ModelT Model type (e.g., LLMModel, ASRModel, TTSModel)
 */
template <typename ModelT>
class ModelFactory {
 public:
  /// Creator function type
  using Creator = std::function<std::unique_ptr<ModelT>(const ModelConfig&)>;

  /// Registry entry
  struct RegistryEntry {
    std::string name;         // Model name
    ModelSeries series;       // Model series
    std::string description;  // Description
    Creator creator;          // Creator function
  };

  /**
   * @brief Register a model type
   * @param name Model name
   * @param series Model series
   * @param creator Creator function
   * @param description Optional description
   */
  static void Register(const std::string& name,
                       ModelSeries series,
                       Creator creator,
                       const std::string& description = "") {
    std::lock_guard<std::mutex> lock(GetMutex());

    RegistryEntry entry{name, series, description, creator};
    GetRegistry()[name] = entry;

    std::cout << "[ModelFactory] Registered model: " << name;
    if (!description.empty()) {
      std::cout << " - " << description;
    }
    std::cout << std::endl;
  }

  /**
   * @brief Create model by series
   * @param series Model series
   * @param config Model configuration
   * @return Model instance, or nullptr if not found
   */
  static std::unique_ptr<ModelT> Create(ModelSeries series,
                                        const ModelConfig& config) {
    std::lock_guard<std::mutex> lock(GetMutex());

    const auto& registry = GetRegistry();

    // Find model matching the series
    for (const auto& [name, entry] : registry) {
      if (entry.series == series) {
        std::cout << "[ModelFactory] Creating model: " << name << std::endl;
        return entry.creator(config);
      }
    }

    std::cerr << "[ModelFactory] Error: No model registered for series: "
              << ModelSeriesToString(series) << std::endl;
    return nullptr;
  }

  /**
   * @brief Create model by name
   * @param name Model name
   * @param config Model configuration
   * @return Model instance, or nullptr if not found
   */
  static std::unique_ptr<ModelT> Create(const std::string& name,
                                        const ModelConfig& config) {
    std::lock_guard<std::mutex> lock(GetMutex());

    const auto& registry = GetRegistry();

    auto it = registry.find(name);
    if (it != registry.end()) {
      std::cout << "[ModelFactory] Creating model: " << name << std::endl;
      return it->second.creator(config);
    }

    std::cerr << "[ModelFactory] Error: No model registered with name: " << name
              << std::endl;
    return nullptr;
  }

  /**
   * @brief List all registered model types
   * @return List of model names
   */
  static std::vector<std::string> ListRegisteredTypes() {
    std::lock_guard<std::mutex> lock(GetMutex());

    std::vector<std::string> types;
    for (const auto& [name, _] : GetRegistry()) {
      types.push_back(name);
    }
    return types;
  }

  /**
   * @brief Get detailed info of all registered models
   * @return List of registry entries
   */
  static std::vector<RegistryEntry> GetRegisteredModels() {
    std::lock_guard<std::mutex> lock(GetMutex());

    std::vector<RegistryEntry> entries;
    for (const auto& [_, entry] : GetRegistry()) {
      entries.push_back(entry);
    }
    return entries;
  }

  /**
   * @brief Check if a series is registered
   * @param series Model series
   * @return Whether registered
   */
  static bool IsRegistered(ModelSeries series) {
    std::lock_guard<std::mutex> lock(GetMutex());

    const auto& registry = GetRegistry();
    for (const auto& [_, entry] : registry) {
      if (entry.series == series) {
        return true;
      }
    }
    return false;
  }

  /**
   * @brief Check if a name is registered
   * @param name Model name
   * @return Whether registered
   */
  static bool IsRegistered(const std::string& name) {
    std::lock_guard<std::mutex> lock(GetMutex());
    return GetRegistry().find(name) != GetRegistry().end();
  }

 private:
  // Get registry (singleton per ModelT)
  static std::map<std::string, RegistryEntry>& GetRegistry() {
    static std::map<std::string, RegistryEntry> registry;
    return registry;
  }

  // Get mutex (singleton per ModelT)
  static std::mutex& GetMutex() {
    static std::mutex mutex;
    return mutex;
  }
};

// ============================================================================
// Model Registrar
// ============================================================================

/**
 * @brief Model auto-registrar for static registration
 *
 * Used with REGISTER_MODEL macro to register models at program startup.
 *
 * @tparam ModelT Model type
 */
template <typename ModelT>
class ModelRegistrar {
 public:
  ModelRegistrar(const std::string& name,
                 ModelSeries series,
                 typename ModelFactory<ModelT>::Creator creator,
                 const std::string& description = "") {
    ModelFactory<ModelT>::Register(name, series, creator, description);
  }
};

// ============================================================================
// Registration Macros
// ============================================================================

/**
 * @brief Macro to register a model
 *
 * This is the unified registration macro for all model types.
 *
 * Example:
 * ```cpp
 * // Register an LLM model
 * REGISTER_MODEL(LLMModel, qwen3_llm, ModelSeries::kQwen3LLM,
 *     [](const ModelConfig& c) { return std::make_unique<Qwen3LLMModel>(c); },
 *     "Qwen3 LLM");
 *
 * // Register an ASR model
 * REGISTER_MODEL(ASRModel, whisper_asr, ModelSeries::kWhisperASR,
 *     [](const ModelConfig& c) { return std::make_unique<WhisperModel>(c); },
 *     "Whisper ASR model");
 * ```
 */
#define REGISTER_MODEL(ModelT, name, series, creator, description)            \
  namespace {                                                                \
  __attribute__((used))                                                      \
  ::houmo::ModelRegistrar<::houmo::ModelT> registrar_##name(#name, series,   \
                                                            creator,         \
                                                            description);    \
  }

}  // namespace houmo
