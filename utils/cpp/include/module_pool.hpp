/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: module_pool.hpp
 * Description:
 *   Module Pool Header File - Defines the ModulePool and PooledModule classes
 * for managing and pooling model modules for efficient inference operations.
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
#ifndef __COMMON_CPP_INCLUDE_MODULE_POOL_HPP__
#define __COMMON_CPP_INCLUDE_MODULE_POOL_HPP__

#include <unistd.h>

#include <cassert>
#include <condition_variable>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "logging.h"
#include "tcim/tcim_runtime.h"
#include "utils.hpp"

namespace fs = std::filesystem;

/* Internal use only */

typedef struct InferTask {
  std::string model_path;
  tcim::Module* module;
  std::map<std::string, tcim::Tensor> inputs;
  std::map<std::string, tcim::Tensor> outputs;
  tcim::Module::RunOption option;
  std::shared_ptr<std::mutex> mutex;
  std::shared_ptr<std::condition_variable> cv;
  bool is_end = false;
  int infer_stage = 0;
  int32_t* stats_vals = nullptr;

  InferTask() : is_end(true) {}

  InferTask(const std::string& model_path, tcim::Module* m,
            std::map<std::string, tcim::Tensor> inputs,
            std::map<std::string, tcim::Tensor> outputs, int infer_stage,
            const tcim::Module::RunOption& opt)
      : model_path(model_path),
        module(m),
        inputs(inputs),
        outputs(outputs),
        infer_stage(infer_stage),
        option(opt) {
    mutex = std::make_shared<std::mutex>();
    cv = std::make_shared<std::condition_variable>();
    stats_vals = new int32_t[3];
    stats_vals[0] = 0;
    stats_vals[1] = 0;
    stats_vals[2] = 0;
  }
} InferTask;

typedef struct InferTaskQueue {
  std::queue<InferTask> queue;
  std::shared_ptr<std::mutex> mutex;
  std::shared_ptr<std::condition_variable> cv;

  InferTaskQueue() {
    mutex = std::make_shared<std::mutex>();
    cv = std::make_shared<std::condition_variable>();
  }
} InferTaskQueue;

typedef struct ModuleExec {
  std::queue<tcim::Module*> queue;
  std::shared_ptr<std::mutex> mutex;
  std::shared_ptr<std::condition_variable> cv;

  ModuleExec() {
    mutex = std::make_shared<std::mutex>();
    cv = std::make_shared<std::condition_variable>();
  }
} ModuleExec;

typedef struct MdInferStats {
  double total_infer_time = 0;
  int64_t infer_num = 0;
  float avg_infer_time = 0.0f;
  float max_infer_time = 0.0f;
  float min_infer_time = 0.0f;
} MdInferStats;

typedef struct StreamStats {
  int64_t infer_task_num;
} StreamStats;

/* Interface structures */

typedef struct PooledMdStats {
  // the number of loaded modules
  int32_t module_num = 0;
  // the number of model inferences
  int64_t infer_num = 0;
  // the average inference time of the module
  float avg_infer_time = 0.0f;
  // the maximum inference time of the module
  float max_infer_time = 0.0f;
  // the minimum inference time of the module
  float min_infer_time = 0.0f;
} PooledMdStats;

typedef struct ModulePoolStats {
  // the number of module types
  size_t module_type_num;
  // key: the names of each type of module
  // value: the number of loaded modules
  std::map<std::string, int32_t> modules_map;
  // the number of streams
  int32_t stream_num;
  // key: stream id
  // value: the number of tasks for each stream inference
  std::map<int32_t, size_t> streams_map;
  // the current number of tasks to be inferred
  size_t infer_task_num;
} ModulePoolStats;

class PooledModule;

class ModulePool {
 public:
  /**
   * @brief ModulePool destructor.
   */
  inline ~ModulePool();
  /**
   * @brief Init module pool, only effective on the first call.
   * @param max_num The maximum number of loaded models.
   * @return ModulePool pointer
   */
  static inline ModulePool* Init(int32_t max_num, int32_t stream_num);
  /**
   * @brief Load model from buffer and generate a pooled module.
   * @param module_name model name, used to identify the model.
   * @param model_data The data buffer of the binary model file.
   * @param len The buffer length.
   * @param option The configuration options for loading model.
   * @return PooledModule pointer
   */
  inline PooledModule* Load(
      const std::string& module_name, const void* model_data, int len,
      const tcim::Module::Option& option = tcim::Module::Option());
  /**
   * @brief Load model from the binary model file and generate a pooled module.
   * @param model_path The file name and path of the binary model file.
   * @param option The configuration options for loading model.
   * @return PooledModule pointer
   */
  inline PooledModule* Load(
      const std::string& model_path,
      const tcim::Module::Option& option = tcim::Module::Option());
  /**
   * @brief Get the statistical information of ModulePool.
   * @param is_print Print statistical information or not, default: not print.
   * @return A structure containing statistical information.
   */
  inline ModulePoolStats GetStats(bool is_print = false);
  /**
   * @brief Get the number of loaded models of the specified model.
   * @param module_name model name, used to specify the model.
   * @return The number of loaded models.
   */
  inline int32_t GetLoadedModuleNum(const std::string& module_name);
  /**
   * @brief Get the inference statistics information of the specified model.
   * Each time this function is called, the existing statistical data will be
   * cleared and the statistics will start anew.
   * @param module_name model name, used to specify the model.
   * @return A structure containing inference statistical information.
   */
  inline MdInferStats GetModuleInferStats(const std::string& module_name);

  /**
   * Internal use only
   * @brief The PooledModule object updates the statistics to ModulePool.
   */
  inline int UpdateInferStats(const std::string& module_name,
                              const float& infer_time);

 private:
  ModulePool() = default;
  ModulePool(const ModulePool&) = delete;
  ModulePool& operator=(const ModulePool&) = delete;

  inline int LoadModule(bool is_file, const std::string& module_name,
                        const tcim::Module::Option& option,
                        const void* model_data = nullptr, const int& len = 0);
  static inline void InferThread(int stream_id,
                                 std::shared_ptr<InferTaskQueue> qin);

  std::mutex module_mutex_;
  std::mutex infer_stats_mutex_;
  std::map<std::string, std::vector<tcim::Module*>> module_map_;
  std::map<std::string, MdInferStats*> infer_stats_map_;

  static inline ModulePool* module_pool_ = nullptr;
  static inline std::once_flag flag_;
  static inline int32_t module_max_num_;
  static inline std::shared_ptr<InferTaskQueue> infer_queue_ = nullptr;
  static inline std::vector<std::thread> threads_;  // infer threads
  static inline std::vector<tcim::Stream> streams_;
  static inline std::mutex module_manager_mutex_;
  static inline std::map<std::string, ModuleExec*> module_manager_;
  static inline std::mutex stream_stats_mutex_;
  static inline std::map<int32_t, StreamStats> stream_stats_map_;
};

class PooledModule {
 public:
  /**
   * @brief PooledModule constructor.
   * @param model_path Model name or path, used to identify the model.
   * @param module The object of the loaded model.
   * @param module_pool The pointer of ModulePool.
   * @param queue Inference task queue.
   */
  inline PooledModule(const std::string& model_path, tcim::Module* module,
                      ModulePool* module_pool,
                      std::shared_ptr<InferTaskQueue> queue);
  /**
   * @brief Gets the total number of input tensors in the model.
   * @return The total number of input tensors.
   */
  inline size_t GetInputNum();
  /**
   * @brief Gets the name of the index‑th input tensor.
   * @param index The position of the input tensor in the model to query for.
   * @return The name of the index‑th input tensor.
   */
  inline std::string GetInputName(int index);
  /**
   * @brief Gets the tensor information, such as tensor shape, data type, and
   *        format with the given input tensor name.
   * @param name The name of the input tensor to query for.
   * @param as_contiguous Updated memory layout information to contiguous or
   * not. If true, equal to: module.GetInputInfo(tensor_name).AsContiguous().
   * @return The tensor information of the tensor.
   */
  inline tcim::TensorInfo GetInputInfo(const std::string& name,
                                       bool as_contiguous);
  /**
   * @brief Gets input tensor on Houmo device with the given tensor name.
   * @param name The name of the input tensor to query for.
   * @return The input tensor.
   */
  inline tcim::Tensor GetInput(const std::string& name);
  /**
   * @brief Gets input data from pre‑allocated memory on host or Houmo device
   * with the given tensor name. The input data includes input tensors defined
   * by the Tensor class.
   * @param name The name of the input tensor to query for.
   * @param tensor The input tensor.
   * @return The status of the function call.
   */
  inline int GetInput(const std::string& name, tcim::Tensor& tensor);
  /**
   * @brief Gets the total number of output tensors in the model.
   * @return The total number of output tensors.
   */
  inline size_t GetOutputNum();
  /**
   * @brief Gets the name of the index‑th output tensor.
   * @param index The position of the output tensor in the model to query for.
   * @return The name of the index‑th output tensor.
   */
  inline std::string GetOutputName(int index);
  /**
   * @brief Gets the information about the output tensor of the model inference,
   * such as tensor shape, data type, and format with the given output tensor
   * name.
   * @param name The name of the output tensor.
   * @param as_contiguous Updated memory layout information to contiguous or
   * not. If true, equal to: module.GetOutputInfo(tensor_name).AsContiguous().
   * @return The information of the output tensor.
   */
  inline tcim::TensorInfo GetOutputInfo(const std::string& name,
                                        bool as_contiguous);
  /**
   * @brief Retrieves the the custom information embeded in the model at model
   * compilation with the custom_msg parameter.
   * @return The custom information string.
   */
  inline std::string GetCustomMsg();
  /**
   * @brief Use the provided inputs to infer the model, and then place the
   * inference result into the given outputs.
   * @param inputs The data to be infered.
   * @param outputs The pre‑allocated memory for storing the inference
   * results.
   * @param option The configuration options for model inference.
   * @return The status of the function call.
   */
  inline tcim::Status Infer(
      const std::map<std::string, tcim::Tensor>& inputs,
      std::map<std::string, tcim::Tensor>& outputs,
      std::vector<int32_t>& perf_stats, int infer_stage = 0,
      const tcim::Module::RunOption& option = tcim::Module::RunOption());
  /**
   * @brief Get the statistical information of the current PooledModule object.
   * @param is_print Print statistical information or not, default: not print.
   * @return A structure containing statistical information.
   */
  inline PooledMdStats GetStats(bool is_print = false);
  /**
   * @brief Get the model name of the current PooledModule object.
   * @return The model name.
   */
  inline std::string GetPooledMdName();

 private:
  inline bool CheckModulePool();
  inline bool CheckTensorsDevice(
      const std::map<std::string, tcim::Tensor>& tensors);

  std::string model_name_ = "";
  tcim::Module* module_ = nullptr;
  ModulePool* module_pool_ = nullptr;
  std::shared_ptr<InferTaskQueue> task_queue_ = nullptr;
};

PooledModule::PooledModule(const std::string& model_path, tcim::Module* module,
                           ModulePool* module_pool,
                           std::shared_ptr<InferTaskQueue> queue)
    : model_name_(model_path),
      module_(module),
      module_pool_(module_pool),
      task_queue_(queue) {
  LOG_DEBUG("Create a PooledModule instance, model name {}", model_name_);
}

size_t PooledModule::GetInputNum() { return module_->GetInputNum(); }

std::string PooledModule::GetInputName(int index) {
  return module_->GetInputName(index);
}

tcim::TensorInfo PooledModule::GetInputInfo(const std::string& name,
                                            bool as_contiguous) {
  if (as_contiguous) {
    return module_->GetInputInfo(name).AsContiguous();
  }
  return module_->GetInputInfo(name);
}

tcim::Tensor PooledModule::GetInput(const std::string& name) {
  return module_->GetInput(name);
}

int PooledModule::GetInput(const std::string& name, tcim::Tensor& tensor) {
  int ret = module_->GetInput(name, tensor);
  return ret;
}

size_t PooledModule::GetOutputNum() { return module_->GetOutputNum(); }

std::string PooledModule::GetOutputName(int index) {
  return module_->GetOutputName(index);
}

tcim::TensorInfo PooledModule::GetOutputInfo(const std::string& name,
                                             bool as_contiguous) {
  if (as_contiguous) {
    return module_->GetOutputInfo(name).AsContiguous();
  }
  return module_->GetOutputInfo(name);
}

std::string PooledModule::GetCustomMsg() { return module_->GetCustomMsg(); }

tcim::Status PooledModule::Infer(
    const std::map<std::string, tcim::Tensor>& inputs,
    std::map<std::string, tcim::Tensor>& outputs,
    std::vector<int32_t>& perf_stats, int infer_stage,
    const tcim::Module::RunOption& option) {
  if (CheckModulePool()) {
    LOG_ERROR("Please initialize the ModulePool first.");
    return tcim::Status::UNINITIALIZED;
  }
  if ((infer_stage < 2) && !CheckTensorsDevice(inputs)) {
    LOG_ERROR("The tensor device in the inputs must be the same.");
    return tcim::Status::INVALID_ARGUMENT;
  }
  if ((infer_stage == 0 || infer_stage == 2) && !CheckTensorsDevice(outputs)) {
    LOG_ERROR("The tensor device in the outputs must be the same.");
    return tcim::Status::INVALID_ARGUMENT;
  }

  InferTask task(model_name_, module_, inputs, outputs, infer_stage, option);

  auto start = GET_TIME();
  std::unique_lock<std::mutex> run_lock(*task.mutex);
  std::unique_lock<std::mutex> queue_lock(*(task_queue_->mutex));
  LOG_DEBUG("==> PooledModule ({}){} push task into task queue.",
            reinterpret_cast<void*>(this), model_name_);
  task_queue_->queue.push(task);
  task_queue_->cv->notify_one();
  queue_lock.unlock();
  LOG_DEBUG("PooledModule ({}){} is waiting task done...",
            reinterpret_cast<void*>(this), model_name_);
  task.cv->wait(run_lock);
  auto end = GET_TIME();
  auto cost = GET_COST(start, end) / 1000.0;

  LOG_DEBUG(
      "PooledModule ({}){} done, infer stage: {}, time cost (ipt/infer/opt): "
      "{} / {} / {} ms.",
      reinterpret_cast<void*>(this), model_name_, infer_stage,
      task.stats_vals[0], task.stats_vals[1], task.stats_vals[2]);
  perf_stats.push_back(task.stats_vals[0]);
  perf_stats.push_back(task.stats_vals[1]);
  perf_stats.push_back(task.stats_vals[2]);

  module_pool_->UpdateInferStats(model_name_, cost);

  delete[] task.stats_vals;
  task.stats_vals = nullptr;

  return tcim::Status::OK;
}

PooledMdStats PooledModule::GetStats(bool is_print) {
  PooledMdStats stats = {0};
  if (CheckModulePool()) {
    LOG_ERROR("Failed to get pooled module {} stats.", model_name_);
    return stats;
  }

  stats.module_num = module_pool_->GetLoadedModuleNum(model_name_);
  auto infer_stats = module_pool_->GetModuleInferStats(model_name_);
  stats.infer_num = infer_stats.infer_num;
  stats.avg_infer_time = infer_stats.avg_infer_time;
  stats.max_infer_time = infer_stats.max_infer_time;
  stats.min_infer_time = infer_stats.min_infer_time;
  if (is_print) {
    LOG_INFO(
        "PooledModule ({}){} stats info, loaded module num:{}, infer num:{}, "
        "infer time (avg/max/min): {}/{}/{} ms.",
        reinterpret_cast<void*>(this), model_name_, stats.module_num,
        stats.infer_num, stats.avg_infer_time, stats.max_infer_time,
        stats.min_infer_time);
  }

  return stats;
}

std::string PooledModule::GetPooledMdName() { return model_name_; }

bool PooledModule::CheckModulePool() { return module_pool_ == nullptr; }

bool PooledModule::CheckTensorsDevice(
    const std::map<std::string, tcim::Tensor>& tensors) {
  if (tensors.empty()) {
    LOG_ERROR("Tensor map is empty.");
    return false;
  }

  tcim::Device device;
  int idx = 0;
  for (const auto& pair : tensors) {
    if (idx == 0) {
      device = pair.second.Device();
    } else if (device != pair.second.Device()) {
      return false;
    }
    idx++;
  }

  return true;
}

ModulePool::~ModulePool() {
  LOG_INFO("===> ModulePool Deinit start.");
  std::unique_lock<std::mutex> queue_lock(*(infer_queue_->mutex));
  InferTask end_task;
  infer_queue_->queue.push(end_task);
  infer_queue_->cv->notify_all();
  queue_lock.unlock();
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i].join();
  }

  for (auto& pair : infer_stats_map_) {
    delete pair.second;
  }
  LOG_DEBUG("===> ModulePool Deinit ready to release module.");

  for (auto& pair : module_map_) {
    for (auto& module : pair.second) {
      delete module;
    }
  }
  LOG_DEBUG("===> ModulePool Deinit after releasing module.");
  infer_stats_map_.clear();
  module_map_.clear();
  streams_.clear();
  module_pool_ = nullptr;
  infer_queue_ = nullptr;
}

ModulePool* ModulePool::Init(int32_t max_num, int32_t stream_num) {
  std::call_once(flag_, [max_num, stream_num]() {
    module_pool_ = new ModulePool();
    infer_queue_ = std::shared_ptr<InferTaskQueue>(new InferTaskQueue());
    streams_.resize(stream_num);
    module_max_num_ = max_num > 0 ? max_num : 1;
    for (int i = 0; i < stream_num; i++) {
      threads_.push_back(
          std::thread(&ModulePool::InferThread, i, infer_queue_));
    }
  });
  LOG_INFO(
      "Init ModulePool {}, the maximum number of modules is {}, the number of "
      "stream is {}.",
      reinterpret_cast<void*>(module_pool_), module_max_num_, stream_num);
  return module_pool_;
}

PooledModule* ModulePool::Load(const std::string& module_name,
                               const void* model_data, int len,
                               const tcim::Module::Option& option) {
  if (model_data == nullptr || len <= 0) {
    LOG_ERROR("Invalid model data or length!");
    return nullptr;
  }

  auto ret = LoadModule(false, module_name, option, model_data, len);
  if (ret != tcim::Status::OK) {
    LOG_ERROR("Load model ({}){} failed, length is {}.", model_data,
              module_name, len);
    return nullptr;
  }

  auto tmp_ptr = module_map_[module_name].back();
  PooledModule* pooled_md = new PooledModule(
      module_name, module_map_[module_name].back(), module_pool_, infer_queue_);

  return pooled_md;
}

PooledModule* ModulePool::Load(const std::string& model_path,
                               const tcim::Module::Option& option) {
  if (!fs::exists(model_path)) {
    LOG_ERROR("{} doesn't exist!", model_path);
    return nullptr;
  }

  auto ret = LoadModule(true, model_path, option);
  if (ret != tcim::Status::OK) {
    LOG_ERROR("Load model {} failed.", model_path);
    return nullptr;
  }

  auto tmp_ptr = module_map_[model_path].back();
  PooledModule* pooled_md = new PooledModule(
      model_path, module_map_[model_path].back(), module_pool_, infer_queue_);

  return pooled_md;
}

int32_t ModulePool::GetLoadedModuleNum(const std::string& module_name) {
  std::lock_guard<std::mutex> module_lock(module_mutex_);
  if (module_map_.find(module_name) == module_map_.end()) {
    return 0;
  }
  return module_map_[module_name].size();
}

MdInferStats ModulePool::GetModuleInferStats(const std::string& module_name) {
  MdInferStats return_stats = {0};

  std::lock_guard<std::mutex> infer_stats_lock(infer_stats_mutex_);
  if (infer_stats_map_.find(module_name) == infer_stats_map_.end()) {
    LOG_ERROR("Invalid model name, return null stats.");
    return return_stats;
  }
  auto infer_stats = infer_stats_map_[module_name];
  return_stats = *infer_stats;

  MdInferStats tmp_struct = {0};
  *infer_stats_map_[module_name] = tmp_struct;

  return return_stats;
}

ModulePoolStats ModulePool::GetStats(bool is_print) {
  ModulePoolStats stats = {0};
  stats.stream_num = streams_.size();
  {
    std::lock_guard<std::mutex> module_map_lock(module_mutex_);
    stats.module_type_num = module_map_.size();
    stats.modules_map = std::map<std::string, int32_t>();
    for (const auto& pair : module_map_) {
      stats.modules_map[pair.first] = pair.second.size();
    }
  }
  {
    std::lock_guard<std::mutex> queue_lock(*(infer_queue_->mutex));
    stats.infer_task_num = infer_queue_->queue.size();
  }
  {
    std::lock_guard<std::mutex> stream_stats_lock(stream_stats_mutex_);
    stats.streams_map = std::map<int32_t, size_t>();
    for (const auto& pair : stream_stats_map_) {
      stats.streams_map[pair.first] =
          stream_stats_map_[pair.first].infer_task_num;
    }
  }

  if (is_print) {
    LOG_INFO(
        "ModulePool {} stats, module_type_num:{}, stream num:{}, "
        "infer_task_num:{}.",
        reinterpret_cast<void*>(this), stats.module_type_num, stats.stream_num,
        stats.infer_task_num);
    for (const auto& pair : stats.modules_map) {
      LOG_INFO("  module name:{}, loaded module num:{}", pair.first,
               pair.second);
    }
    for (const auto& pair : stats.streams_map) {
      LOG_INFO("  stream id:{}, infer task num:{}", pair.first, pair.second);
    }
  }

  return stats;
}

int ModulePool::UpdateInferStats(const std::string& module_name,
                                 const float& infer_time) {
  std::lock_guard<std::mutex> infer_stats_lock(infer_stats_mutex_);
  if (infer_stats_map_.find(module_name) == infer_stats_map_.end()) {
    return -1;
  }

  auto infer_stats = infer_stats_map_[module_name];
  infer_stats->total_infer_time += infer_time;
  infer_stats->infer_num += 1;
  infer_stats->avg_infer_time =
      (1.0f * infer_stats->total_infer_time) / infer_stats->infer_num;
  infer_stats->max_infer_time = infer_time > infer_stats->max_infer_time
                                    ? infer_time
                                    : infer_stats->max_infer_time;
  if (infer_stats->min_infer_time <= 0) {
    infer_stats->min_infer_time = infer_time;
  } else {
    infer_stats->min_infer_time = infer_time < infer_stats->min_infer_time
                                      ? infer_time
                                      : infer_stats->min_infer_time;
  }

  return 0;
}

int ModulePool::LoadModule(bool is_file, const std::string& module_name,
                           const tcim::Module::Option& option,
                           const void* model_data, const int& len) {
  std::lock_guard<std::mutex> module_lock(module_mutex_);

  int module_num = module_map_[module_name].size();
  if (module_num < module_max_num_) {
    if (module_num == 0) {
      // the first time to load model
      {
        std::lock_guard<std::mutex> infer_stats_lock(infer_stats_mutex_);
        infer_stats_map_[module_name] = new MdInferStats();
      }
    }
    tcim::Status ret;
    tcim::Module* module = new tcim::Module();
    if (is_file) {
      ret = module->LoadModel(module_name, option);
    } else {
      ret = module->LoadModel(model_data, len, option);
    }
    if (ret != tcim::Status::OK) {
      return ret;
    }
    module_map_[module_name].push_back(module);

    {
      std::lock_guard<std::mutex> module_manager_lock(module_manager_mutex_);
      if (module_manager_.find(module_name) == module_manager_.end()) {
        module_manager_[module_name] = new ModuleExec();
      }
      {
        std::lock_guard<std::mutex> module_task_lock(
            *(module_manager_[module_name]->mutex));
        module_manager_[module_name]->queue.push(module);
      }
    }
    LOG_DEBUG("Add module {} into pooled module, size:{}", module_name,
              module_map_[module_name].size());
  }

  return tcim::Status::OK;
}

void ModulePool::InferThread(int stream_id,
                             std::shared_ptr<InferTaskQueue> qin) {
  LOG_DEBUG("===> ModulePool InferThread start, stream id:{}", stream_id);
  auto start = GET_TIME();
  auto end = GET_TIME();
  tcim::Stream stream = streams_[stream_id];
  StreamStats stream_stats = {0};
  {
    std::lock_guard<std::mutex> stream_stats_lock(stream_stats_mutex_);
    stream_stats_map_[stream_id] = stream_stats;
  }
  while (true) {
    std::unique_lock<std::mutex> queue_lock(*(qin->mutex));
    while (qin->queue.empty()) {
      qin->cv->wait(queue_lock);
    }
    auto task = qin->queue.front();
    if (task.is_end) {
      queue_lock.unlock();
      break;
    }
    qin->queue.pop();
    queue_lock.unlock();

    ModuleExec* module_exec;
    {
      std::lock_guard<std::mutex> module_task_lock(module_manager_mutex_);
      module_exec = module_manager_[task.model_path];
      LOG_DEBUG(
          "---> InferThread stream {}, module queue size:{}, module_exec:{}",
          stream_id, module_exec->queue.size(),
          reinterpret_cast<void*>(module_exec));
    }

    tcim::Device outputs_device;
    if (task.infer_stage == 0 || task.infer_stage == 2) {
      outputs_device = task.outputs.begin()->second.Device();
    }

    tcim::Module* module;
    {
      std::unique_lock<std::mutex> module_queue_lock(*(module_exec->mutex));
      while (module_exec->queue.empty()) {
        LOG_DEBUG("---> InferThread stream {} is waiting free module...",
                  stream_id);
        module_exec->cv->wait(module_queue_lock);
      }
      module = module_exec->queue.front();
      module_exec->queue.pop();
      module_queue_lock.unlock();
      LOG_DEBUG(
          "---> InferThread stream {}, get module:{}, module queue size:{}, "
          "outputs device: {}, ready to execute.",
          stream_id, reinterpret_cast<void*>(module), module_exec->queue.size(),
          static_cast<int>(outputs_device));
    }
    module->SetStream(stream);

    // 5. set input
    if (task.infer_stage < 2) {
      LOG_DEBUG("ModulePool InferThread, stream id:{}, set input.", stream_id);
      start = GET_TIME();
      for (const auto& input : task.inputs) {
        module->SetInput(input.first, input.second);
      }
      end = GET_TIME();
      task.stats_vals[0] = GET_COST(start, end);
    }

    if ((task.infer_stage == 0 || task.infer_stage == 2) &&
        (outputs_device == tcim::Device::HDPL)) {
      // 6. set output
      start = GET_TIME();
      for (auto& output : task.outputs) {
        module->SetOutput(output.first, output.second);
      }
      end = GET_TIME();
      task.stats_vals[2] = GET_COST(start, end);
    }

    LOG_DEBUG("InferThread stream {} start inference", stream_id);
    // 7. run and sync
    start = GET_TIME();
    module->Run(false, task.option);
    module->Sync();
    end = GET_TIME();
    task.stats_vals[1] = GET_COST(start, end);

    if ((task.infer_stage == 0 || task.infer_stage == 2) &&
        outputs_device == tcim::Device::CPU) {
      LOG_DEBUG("InferThread stream {} start get output", stream_id);
      // 8. get output
      start = GET_TIME();
      for (auto& output : task.outputs) {
        module->GetOutput(output.first, output.second);
      }
      end = GET_TIME();
      task.stats_vals[2] = GET_COST(start, end);
    }

    std::lock_guard<std::mutex> run_lock(*task.mutex);
    task.cv->notify_one();

    // 9. release module
    {
      std::lock_guard<std::mutex> module_manager_lock(module_manager_mutex_);
      std::lock_guard<std::mutex> module_lock(*(module_exec->mutex));
      module_exec->queue.push(module);
      module_exec->cv->notify_one();
      LOG_DEBUG(
          "---> InferThread stream {} release module {}, module queue size:{}.",
          stream_id, reinterpret_cast<void*>(module),
          module_exec->queue.size());
    }

    {
      std::lock_guard<std::mutex> stream_stats_lock(stream_stats_mutex_);
      stream_stats_map_[stream_id].infer_task_num += 1;
    }
  }
  LOG_DEBUG("<== ModulePool InferThread end, stream id:{}", stream_id);
}

#endif  // __COMMON_CPP_INCLUDE_MODULE_POOL_HPP__
