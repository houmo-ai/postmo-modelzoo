/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: model_factory.cc
 * Description:
 *   Model factory implementation for creating model instances by series or name.
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

#include "core/model_factory.h"
#include "core/llm_model.h"

#include <algorithm>

namespace houmo {

// Get the registry (static local variable, thread-safe in C++11)
std::map<std::string, ModelFactory::RegistryEntry>& ModelFactory::GetRegistry() {
  static std::map<std::string, RegistryEntry> registry;
  return registry;
}

// Get the mutex
std::mutex& ModelFactory::GetMutex() {
  static std::mutex mutex;
  return mutex;
}

void ModelFactory::Register(const std::string& name,
                            ModelSeries series,
                            Creator creator,
                            const std::string& description) {
  std::lock_guard<std::mutex> lock(GetMutex());

  RegistryEntry entry{name, creator, series, description};
  GetRegistry()[name] = entry;

  std::cout << "[ModelFactory] Registered model: " << name;
  if (!description.empty()) {
    std::cout << " - " << description;
  }
  std::cout << std::endl;
}

std::unique_ptr<LLMModel> ModelFactory::Create(ModelSeries series,
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

std::unique_ptr<LLMModel> ModelFactory::Create(const std::string& name,
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

std::vector<std::string> ModelFactory::ListRegisteredTypes() {
  std::lock_guard<std::mutex> lock(GetMutex());

  std::vector<std::string> types;
  for (const auto& [name, _] : GetRegistry()) {
    types.push_back(name);
  }
  std::sort(types.begin(), types.end());
  return types;
}

std::vector<ModelFactory::RegistryEntry> ModelFactory::GetRegisteredModels() {
  std::lock_guard<std::mutex> lock(GetMutex());

  std::vector<RegistryEntry> entries;
  for (const auto& [_, entry] : GetRegistry()) {
    entries.push_back(entry);
  }
  return entries;
}

bool ModelFactory::IsRegistered(ModelSeries series) {
  std::lock_guard<std::mutex> lock(GetMutex());

  const auto& registry = GetRegistry();
  for (const auto& [_, entry] : registry) {
    if (entry.series == series) {
      return true;
    }
  }
  return false;
}

bool ModelFactory::IsRegistered(const std::string& name) {
  std::lock_guard<std::mutex> lock(GetMutex());
  return GetRegistry().find(name) != GetRegistry().end();
}

}  // namespace houmo
