#ifndef _APIS_COMMON_HPP_MODULE_POOL_HPP_
#define _APIS_COMMON_HPP_MODULE_POOL_HPP_

#include <unistd.h>

#include <cassert>
#include <condition_variable>
#include <iostream>
#include <map>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include "logging.h"
#include "tcim/tcim_runtime.h"
#include "utils.hpp"

/* Internal use only */

typedef struct RunTask {
  std::string model_path;
  tcim::Module* module;
  std::map<std::string, tcim::Tensor> inputs;
  std::map<std::string, tcim::Tensor> outputs;
  tcim::Module::RunOption option;
  std::shared_ptr<std::mutex> mutex;
  std::shared_ptr<std::condition_variable> cv;
  bool is_end = false;

  RunTask() : is_end(true) {}

  RunTask(const std::string& model_path, tcim::Module* m,
          std::map<std::string, tcim::Tensor> inputs,
          std::map<std::string, tcim::Tensor> outputs,
          const tcim::Module::RunOption& opt)
      : model_path(model_path),
        module(m),
        inputs(inputs),
        outputs(outputs),
        option(opt) {
    mutex = std::make_shared<std::mutex>();
    cv = std::make_shared<std::condition_variable>();
  }
} RunTask;

typedef struct InferTaskQueue {
  std::queue<RunTask> queue;
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
  int32_t module_num = 0;  // the number of loaded module
  int64_t infer_num = 0;   // the number of module inference
  float avg_infer_time = 0.0f;
  float max_infer_time = 0.0f;
  float min_infer_time = 0.0f;
} PooledMdStats;

typedef struct ModulePoolStats {
  // the number of module type
  size_t module_type_num;
  // the names and inference number of each module
  std::map<std::string, int32_t> modules_map;
  // the number of tcim::stream
  int32_t stream_num;
  // the id and task number of each stream
  std::map<int32_t, size_t> streams_map;
  // the current number of tasks to be inferred
  size_t infer_task_num;
} ModulePoolStats;

class PooledModule;

class ModulePool {
 public:
  ~ModulePool();
  static ModulePool* Init(int32_t max_num);
  PooledModule* Load(
      const std::string& model_name, const void* model_data, int len,
      const tcim::Module::Option& option = tcim::Module::Option());
  PooledModule* Load(
      const std::string& model_path,
      const tcim::Module::Option& option = tcim::Module::Option());
  ModulePoolStats GetStats(bool is_print = false);

  int32_t GetDeviceId(const std::string& module_name);
  int32_t GetLoadedModuleNum(const std::string& module_name);
  MdInferStats GetModuleInferStats(const std::string& module_name);
  int UpdateInferStats(const std::string& module_name, const float& infer_time);

 private:
  ModulePool() = default;
  ModulePool(const ModulePool&) = delete;
  ModulePool& operator=(const ModulePool&) = delete;

  bool CheckDeviceId(const int& device_id);
  int LoadModule(bool is_file, const int& device_id,
                 const std::string& model_name,
                 const tcim::Module::Option& option,
                 const void* model_data = nullptr, const int& len = 0);
  static void InferThread(int stream_id, std::shared_ptr<InferTaskQueue> qin);

  std::mutex module_mutex_;
  std::mutex module_dev_mutex_;
  std::mutex infer_stats_mutex_;
  std::map<std::string, std::vector<tcim::Module*>> module_map_;
  std::map<std::string, tcim::Module::WeightManager> module_wm_map_;
  std::map<std::string, int32_t> module_device_map_;
  std::map<std::string, MdInferStats*> infer_stats_map_;

  static ModulePool* module_pool_;
  static std::once_flag flag_;
  static int32_t module_max_num_;
  static std::shared_ptr<InferTaskQueue> infer_queue_;
  static std::vector<std::thread> threads_;  // infer threads
  static std::vector<tcim::Stream> streams_;
  static std::mutex module_manager_mutex_;
  static std::map<std::string, ModuleExec*> module_manager_;
  static std::mutex stream_stats_mutex_;
  static std::map<int32_t, StreamStats> stream_stats_map_;
};

class PooledModule {
 public:
  PooledModule(const std::string& model_path, tcim::Module* module,
               ModulePool* module_pool, std::shared_ptr<InferTaskQueue> queue);

  size_t GetInputNum();
  std::string GetInputName(int index);
  tcim::TensorInfo GetInputInfo(const std::string& name, bool as_contiguous);
  tcim::Tensor GetInput(const std::string& name);
  int GetInput(const std::string& name, tcim::Tensor& tensor);
  size_t GetOutputNum();
  std::string GetOutputName(int index);
  tcim::TensorInfo GetOutputInfo(const std::string& name, bool as_contiguous);
  tcim::Status Infer(
      const std::map<std::string, tcim::Tensor>& inputs,
      std::map<std::string, tcim::Tensor>& outputs,
      const tcim::Module::RunOption& option = tcim::Module::RunOption());
  PooledMdStats GetStats(bool is_print = false);

 private:
  std::string model_name_;
  tcim::Module* module_;
  ModulePool* module_pool_;
  std::shared_ptr<InferTaskQueue> task_queue_;
};

PooledModule::PooledModule(const std::string& model_path, tcim::Module* module,
                           ModulePool* module_pool,
                           std::shared_ptr<InferTaskQueue> queue)
    : model_name_(model_path),
      module_(module),
      module_pool_(module_pool),
      task_queue_(queue) {
  std::cout << "Create a PooledModule instance, model name" << model_name_
            << std::endl;
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

tcim::Status PooledModule::Infer(
    const std::map<std::string, tcim::Tensor>& inputs,
    std::map<std::string, tcim::Tensor>& outputs,
    const tcim::Module::RunOption& option) {
  RunTask task(model_name_, module_, inputs, outputs, option);

  auto start = GET_TIME();
  std::unique_lock<std::mutex> run_lock(*task.mutex);
  std::unique_lock<std::mutex> queue_lock(*(task_queue_->mutex));
  // std::cout << "==> PooledModule push task into task queue" << std::endl;
  task_queue_->queue.push(task);
  task_queue_->cv->notify_one();
  queue_lock.unlock();
  std::cout << "PooledModule name:" << model_name_ << ", waiting task done..."
            << std::endl;
  task.cv->wait(run_lock);
  auto end = GET_TIME();
  auto cost = GET_COST(start, end) / 1000.0;

  module_pool_->UpdateInferStats(model_name_, cost);

  return tcim::Status::OK;
}

PooledMdStats PooledModule::GetStats(bool is_print) {
  PooledMdStats stats = {0};
  if (module_pool_ == nullptr) {
    std::cerr << "Failed to get pooled module " << model_name_
              << " stats, thread id:" << std::this_thread::get_id()
              << std::endl;
    return stats;
  }

  stats.module_num = module_pool_->GetLoadedModuleNum(model_name_);
  auto infer_stats = module_pool_->GetModuleInferStats(model_name_);
  stats.infer_num = infer_stats.infer_num;
  stats.avg_infer_time = infer_stats.avg_infer_time;
  stats.max_infer_time = infer_stats.max_infer_time;
  stats.min_infer_time = infer_stats.min_infer_time;
  if (is_print) {
    std::cout << "pooled module " << model_name_
              << ", stats info, thread id:" << std::this_thread::get_id()
              << ", loaded module num:" << stats.module_num
              << ", infer num:" << stats.infer_num
              << ", infer time (avg/max/min): " << stats.avg_infer_time << "/"
              << stats.max_infer_time << "/" << stats.min_infer_time << " ms."
              << std::endl;
  }

  return stats;
}

ModulePool* ModulePool::module_pool_ = nullptr;
std::once_flag ModulePool::flag_;
int32_t ModulePool::module_max_num_;
std::shared_ptr<InferTaskQueue> ModulePool::infer_queue_ = nullptr;
std::vector<std::thread> ModulePool::threads_;
std::vector<tcim::Stream> ModulePool::streams_;
std::mutex ModulePool::module_manager_mutex_;
std::map<std::string, ModuleExec*> ModulePool::module_manager_;
std::mutex ModulePool::stream_stats_mutex_;
std::map<int32_t, StreamStats> ModulePool::stream_stats_map_;

ModulePool::~ModulePool() {
  std::cout << "===> ModulePool Deinit start." << std::endl;
  std::unique_lock<std::mutex> queue_lock(*(infer_queue_->mutex));
  RunTask end_task;
  infer_queue_->queue.push(end_task);
  infer_queue_->cv->notify_all();
  queue_lock.unlock();
  for (int i = 0; i < threads_.size(); i++) {
    threads_[i].join();
  }

  for (auto& pair : infer_stats_map_) {
    delete pair.second;
  }

  for (auto& pair : module_map_) {
    for (auto& module : pair.second) {
      delete module;
    }
  }
  infer_stats_map_.clear();
  module_map_.clear();
  streams_.clear();
}

ModulePool* ModulePool::Init(int32_t max_num) {
  std::cout << "===> ModulePool Init start." << std::endl;
  std::call_once(flag_, []() {
    module_pool_ = new ModulePool();
    infer_queue_ = std::shared_ptr<InferTaskQueue>(new InferTaskQueue());
    int stream_num = 8;
    streams_.resize(stream_num);
    for (int i = 0; i < stream_num; i++) {
      threads_.push_back(
          std::thread(&ModulePool::InferThread, i, infer_queue_));
    }
  });
  module_max_num_ = max_num > 0 ? max_num : 1;
  return module_pool_;
}

PooledModule* ModulePool::Load(const std::string& model_name,
                               const void* model_data, int len,
                               const tcim::Module::Option& option) {
  if (model_data == nullptr or len <= 0) {
    std::cerr << "Invalid model data or length!" << std::endl;
    return nullptr;
  }

  int device_id = option.device_id;
  if (device_id < 0) {
    std::cerr << "Invalid module option, device id is " << device_id
              << std::endl;
    return nullptr;
  }

  auto ret = LoadModule(false, device_id, model_name, option, model_data, len);
  if (ret != tcim::Status::OK) {
    std::cerr << "load model " << model_data << " length " << len << " failed."
              << std::endl;
    return nullptr;
  }

  auto tmp_ptr = module_map_[model_name].back();
  PooledModule* pooled_md = new PooledModule(
      model_name, module_map_[model_name].back(), module_pool_, infer_queue_);

  return pooled_md;
}

PooledModule* ModulePool::Load(const std::string& model_path,
                               const tcim::Module::Option& option) {
  if (!fs::exists(model_path)) {
    std::cerr << model_path << " not exist!" << std::endl;
    return nullptr;
  }

  int device_id = option.device_id;
  if (device_id < 0) {
    std::cerr << "Invalid module option, device id is " << device_id
              << std::endl;
    return nullptr;
  }

  auto ret = LoadModule(true, device_id, model_path, option);
  if (ret != tcim::Status::OK) {
    std::cerr << "load model " << model_path << " failed." << std::endl;
    return nullptr;
  }

  auto tmp_ptr = module_map_[model_path].back();
  PooledModule* pooled_md = new PooledModule(
      model_path, module_map_[model_path].back(), module_pool_, infer_queue_);

  return pooled_md;
}

int32_t ModulePool::GetDeviceId(const std::string& module_name) {
  std::lock_guard<std::mutex> module_dev_lock(module_dev_mutex_);
  if (module_device_map_.find(module_name) == module_device_map_.end()) {
    return -1;
  }
  return module_device_map_[module_name];
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
    std::cerr << "GetModuleInferStats, return null stats." << std::endl;
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
    std::cout << "ModulePool stats, module_type_num:" << stats.module_type_num
              << ", stream num:" << stats.stream_num
              << ", infer_task_num:" << stats.infer_task_num << std::endl;
    for (const auto& pair : stats.modules_map) {
      std::cout << "  module name:" << pair.first
                << ", loaded module num:" << pair.second << std::endl;
    }
    for (const auto& pair : stats.streams_map) {
      std::cout << "  stream id:" << pair.first
                << ", infer task num:" << pair.second << std::endl;
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

bool ModulePool::CheckDeviceId(const int& device_id) {
  std::lock_guard<std::mutex> module_dev_lock(module_dev_mutex_);
  if (module_device_map_.empty()) {
    return true;
  }
  for (const auto& pair : module_device_map_) {
    if (pair.second != device_id) {
      return false;
    }
  }
  return true;
}

int ModulePool::LoadModule(bool is_file, const int& device_id,
                           const std::string& model_name,
                           const tcim::Module::Option& option,
                           const void* model_data, const int& len) {
  std::lock_guard<std::mutex> module_lock(module_mutex_);

  if (!CheckDeviceId(device_id)) {
    std::cerr << "Invalid device id " << device_id << std::endl;
    return -1;
  }

  int module_num = module_map_[model_name].size();
  if (module_num < module_max_num_) {
    tcim::Module::WeightManager wm;
    if (module_num == 0) {
      // the first time to load model
      wm = tcim::Module::WeightManager::CreateWeightManager(device_id);
      module_wm_map_[model_name] = wm;
      {
        std::lock_guard<std::mutex> module_dev_lock(module_dev_mutex_);
        module_device_map_[model_name] = device_id;
      }
      {
        std::lock_guard<std::mutex> infer_stats_lock(infer_stats_mutex_);
        infer_stats_map_[model_name] = new MdInferStats();
      }
    } else {
      wm = module_wm_map_[model_name];
    }
    tcim::Module::Option option_new(device_id);
    option_new = option;
    option_new.weight_manager = wm;
    tcim::Status ret;
    tcim::Module* module = new tcim::Module();
    if (is_file) {
      ret = module->LoadModel(model_name, option_new);
    } else {
      ret = module->LoadModel(model_data, len, option_new);
    }
    if (ret != tcim::Status::OK) {
      return ret;
    }
    module_map_[model_name].push_back(module);

    {
      std::lock_guard<std::mutex> module_manager_lock(module_manager_mutex_);
      if (module_manager_.find(model_name) == module_manager_.end()) {
        module_manager_[model_name] = new ModuleExec();
      }
      {
        std::lock_guard<std::mutex> module_task_lock(
            *(module_manager_[model_name]->mutex));
        module_manager_[model_name]->queue.push(module);
      }
    }
    std::cout << "Add module " << model_name
              << " into pooled module, size:" << module_map_[model_name].size()
              << std::endl;
  }

  return tcim::Status::OK;
}

void ModulePool::InferThread(int stream_id,
                             std::shared_ptr<InferTaskQueue> qin) {
  std::cout << "===> ModulePool InferThread start, stream id:" << stream_id
            << std::endl;
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
      // std::cout << "===> ModulePool InferThread stream id:" << stream_id
      //           << ", module queue size:" << module_exec->queue.size()
      //           << ", module_exec:" << reinterpret_cast<void*>(module_exec)
      //           << std::endl;
    }

    tcim::Module* module;
    {
      std::unique_lock<std::mutex> module_queue_lock(*(module_exec->mutex));
      while (module_exec->queue.empty()) {
        std::cout << "===> ModulePool InferThread stream id:" << stream_id
                  << ", waiting free module..." << std::endl;
        module_exec->cv->wait(module_queue_lock);
      }
      module = module_exec->queue.front();
      module_exec->queue.pop();
      module_queue_lock.unlock();
      // std::cout << "===> ModulePool InferThread stream id:" << stream_id
      //           << ", get module:" << reinterpret_cast<void*>(module)
      //           << ", module queue size:" << module_exec->queue.size()
      //           << ", ready to execute." << std::endl;
    }
    module->SetStream(stream);

    // 5. set input
    for (const auto& input : task.inputs) {
      module->SetInput(input.first, input.second);
    }

    // 6. run and sync
    module->Run();
    module->Sync();

    // 7. get output
    for (auto& output : task.outputs) {
      module->GetOutput(output.first, output.second);
    }

    std::lock_guard<std::mutex> run_lock(*task.mutex);
    task.cv->notify_one();

    // release module
    {
      std::lock_guard<std::mutex> module_manager_lock(module_manager_mutex_);
      std::lock_guard<std::mutex> module_lock(*(module_exec->mutex));
      module_exec->queue.push(module);
      module_exec->cv->notify_one();
      // std::cout << "===> ModulePool InferThread stream id:" << stream_id
      //           << ", release module:" << reinterpret_cast<void*>(module)
      //           << ", module queue size:" << module_exec->queue.size()
      //           << std::endl;
    }

    {
      std::lock_guard<std::mutex> stream_stats_lock(stream_stats_mutex_);
      stream_stats_map_[stream_id].infer_task_num += 1;
    }
  }
  std::cout << "<== ModulePool InferThread end." << std::endl;
}

#endif  // _APIS_COMMON_HPP_MODULE_POOL_HPP_