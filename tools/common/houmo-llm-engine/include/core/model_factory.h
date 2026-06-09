/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: model_factory.h
 * Description:
 *   Model factory supporting static registration and runtime model type
 *   selection. New models can be registered without modifying framework code.
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

// Forward declaration
class LLMModel;

/**
 * @brief Model series enumeration
 */
enum class ModelSeries {
  kUnknown,     // Unknown type
  kQwen3LLM,   // Qwen3 pure-text LLM
  kQwen35MLLM,  // Qwen3.5 multimodal MLLM
  kQwen3VLM,   // Qwen3-VL vision-language
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
  }
  return ModelSeries::kUnknown;
}

/**
 * @brief Model factory
 *
 * Supports static registration and runtime selection:
 * 1. Register models using REGISTER_LLM_MODEL macro
 * 2. Create instances via Create(series, config)
 */
class ModelFactory {
 public:
  // Model creator function type
  using Creator = std::function<std::unique_ptr<LLMModel>(const ModelConfig&)>;

  // Registry entry
  struct RegistryEntry {
    std::string name;        // Model name
    Creator creator;         // Creator function
    ModelSeries series;      // Model series
    std::string description; // Description
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
                       const std::string& description = "");

  /**
   * @brief Create model by series
   * @param series Model series
   * @param config Model configuration
   * @return Model instance
   */
  static std::unique_ptr<LLMModel> Create(ModelSeries series,
                                          const ModelConfig& config);

  /**
   * @brief Create model by name
   * @param name Model name
   * @param config Model configuration
   * @return Model instance
   */
  static std::unique_ptr<LLMModel> Create(const std::string& name,
                                          const ModelConfig& config);

  /**
   * @brief List all registered model types
   * @return List of model names
   */
  static std::vector<std::string> ListRegisteredTypes();

  /**
   * @brief Get detailed info of all registered models
   * @return List of registry entries
   */
  static std::vector<RegistryEntry> GetRegisteredModels();

  /**
   * @brief Check if a series is registered
   * @param series Model series
   * @return Whether registered
   */
  static bool IsRegistered(ModelSeries series);

  /**
   * @brief Check if a name is registered
   * @param name Model name
   * @return Whether registered
   */
  static bool IsRegistered(const std::string& name);

 private:
  // Get registry (singleton)
  static std::map<std::string, RegistryEntry>& GetRegistry();

  // Get mutex
  static std::mutex& GetMutex();
};

/**
 * @brief Model auto-registrar
 *
 * Used with REGISTER_LLM_MODEL macro to register models at program startup.
 */
class ModelRegistrar {
 public:
  ModelRegistrar(const std::string& name,
                 ModelSeries series,
                 ModelFactory::Creator creator,
                 const std::string& description = "") {
    ModelFactory::Register(name, series, creator, description);
  }
};

/**
 * @brief Macro to register an LLM model
 *
 * Example:
 * ```cpp
 * REGISTER_LLM_MODEL(qwen3_llm, ModelSeries::kQwen3LLM,
 *                    [](const ModelConfig& c) { return std::make_unique<Qwen3LLMModel>(c); },
 *                    "Qwen3 pure-text LLM");
 * ```
 */
#define REGISTER_LLM_MODEL(name, series, creator, description)              \
  namespace {                                                              \
  __attribute__((used))                                                    \
  ::houmo::ModelRegistrar registrar_##name(#name, series, creator,         \
                                           description);                   \
  }

}  // namespace houmo
