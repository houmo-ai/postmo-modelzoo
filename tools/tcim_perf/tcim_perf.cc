/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: tcim_perf.cc
 * Description:
 *   TCIM Performance Testing Tool - Main application for measuring
 * performance of TCIM (Tensor Compiler In Memory) models
 * using multi-threaded execution and various performance metrics.
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
#include <stdio.h>

#include <atomic>
#include <chrono>  // Include time unit definitions
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <future>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <nlohmann/json.hpp>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <typeinfo>
#include <vector>

#if (__GNUC__ < 8) && 0
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#ifdef XH2A_HM_SYS
#ifdef __cplusplus
extern "C" {
#endif
#include "hm_sys.h"
#ifdef __cplusplus
}
#endif
#endif

#include "../common/module_pool.hpp"
#include "../common/npy.hpp"
#include "../common/stream_engine.hpp"
#include "tcim/tcim_runtime.h"

#define COLOR_RED "\x1b[91;20m"
#define COLOR_GREEN "\x1b[92;20m"
#define COLOR_YELLOW "\x1b[93;20m"
#define COLOR_BLUE "\x1b[94;20m"
#define COLOR_MAGENT "\x1b[95;20m"
#define COLOR_CYAN "\x1b[96;20m"
#define COLOR_RESET "\x1b[0m"

#define GET_TIME() std::chrono::system_clock::now()
#define GET_COST(start, end) \
  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()

using json = nlohmann::json;

struct CliArguments {
  std::string model_path;
  std::string model_name;
  std::string input_path = "";
  std::string output_path = ".";
  size_t batch = 1;
  size_t warm_up = 1;
  size_t threads = 1;
  size_t devices = 1;
  size_t loops = 1;
  size_t samples = 1;
  bool infer_only = false;
  bool module_pool = false;
  size_t streams = 4;
  size_t modules = 0;
  size_t interval = 0;
  size_t queue_length = 0;
};

typedef struct {
  std::string model_path;
  tcim::Module::WeightManager weight_manager;
  int loop_num = 0;
  int sample_cnt = 0;
  int warm_up = 0;
  bool infer_only = false;
  bool is_result_check = true;
  uint32_t infer_min_cost = 99999;
  uint32_t infer_max_cost = 0;
  uint32_t infer_total_cost = 0;
  uint32_t input_min_cost = 99999;
  uint32_t input_max_cost = 0;
  uint32_t input_total_cost = 0;
  uint32_t output_min_cost = 99999;
  uint32_t output_max_cost = 0;
  uint32_t output_total_cost = 0;
  uint32_t e2e_min_cost = 99999;
  uint32_t e2e_max_cost = 0;
  uint32_t e2e_total_cost = 0;
} ThreadInfo;

typedef struct {
  uint64_t req_id;
  std::map<std::string, tcim::Tensor> data_in;
  std::map<std::string, tcim::Tensor> data_out;
  std::map<std::string, tcim::Tensor> ref_out;
} Task;

typedef struct {
  std::queue<Task> queue;
  std::mutex mutex;
  std::condition_variable cond;
  // std::map<std::string, tcim::TensorInfo> info_map;
} TaskQueue;

/**
 * @brief whether the file exists
 *
 * @param file_path file path
 * @return true file exists
 * @return false file does not exist
 */
bool IsFileExists(std::string file_path) {
  std::ifstream f(file_path.c_str());
  return f.good();
}

std::string TensorInfo2Str(const tcim::TensorInfo &tensor_info) {
  std::stringstream ss;
  ss << tensor_info;
  return ss.str();
}

/**
 * @brief Parse cmdline arguments to struct *arguments
 *
 * @param arguments pointer to output CliArguments struct
 * @param argc cmdline argument count
 * @param argv cmdline argument char*
 * @return true parse command line succeed
 * @return false parse command line failed
 */
void PrintUsage() {
  std::cout << "Usage:" << std::endl;
  std::cout << "--model/-m: (required) model path" << std::endl;
  std::cout << "--input/-i: input data & output golden folder path"
            << std::endl;
  std::cout << "--warm_up/-w: default 1, the number of warm up times"
            << std::endl;
  std::cout << "--batch/-b: default 1, the batch of model" << std::endl;
  std::cout << "--loops/-l: default 1, the number of loop"
               " iterations for model inference only"
            << std::endl;
  std::cout << "--threads/-t: default 1, the number of threads" << std::endl;
  std::cout << "--devices/-d: default 1, the number of devices" << std::endl;
  std::cout << "--samples/-s: default 1, the number of samples" << std::endl;
  std::cout << "--output/-o: default to the current path, the folder "
               "path where the perf result file (hmperf.txt) is stored."
            << std::endl;
  std::cout << "--infer_only/-y: default false, only execute inference."
            << std::endl;
  std::cout << "--name/-n: model name, used for reading the input data and "
               "output golen of the model."
            << std::endl;
  std::cout << "--module_pool/-p: default false, testing using the module pool."
            << std::endl;
  std::cout << "--streams/-e: default 4, the number of streams." << std::endl;
  std::cout << "--modules/-c: default 1, the maximum number of models "
               "that can be loaded. (Only effective in the module pool)"
            << std::endl;
  std::cout << "--interval/-v: push an input data every <interval> "
               "milliseconds."
            << std::endl;
  std::cout << "--queue_length/-q: the maximum number of tasks in the "
               "task queue. (Only effective when setting the interval param.)"
            << std::endl;
}

bool ParseArgs(CliArguments *arguments, int argc, char *argv[]) {
  static const std::map<std::string, std::pair<char, bool>> opt_map = {
      {"help", {'h', false}},        {"model", {'m', true}},
      {"input", {'i', true}},        {"warm_up", {'w', true}},
      {"batch", {'b', true}},        {"loops", {'l', true}},
      {"threads", {'t', true}},      {"devices", {'d', true}},
      {"samples", {'s', true}},      {"output", {'o', true}},
      {"infer_only", {'y', true}},   {"name", {'n', true}},
      {"module_pool", {'p', true}},  {"streams", {'e', true}},
      {"modules", {'c', true}},      {"interval", {'v', true}},
      {"queue_length", {'q', true}},
  };

  for (int i = 1; i < argc; i++) {
    std::string arg = argv[i];
    char ch = 0;
    std::string val;

    if (arg.compare(0, 2, "--") == 0) {
      std::string name = arg.substr(2);
      auto it = opt_map.find(name);
      if (it == opt_map.end()) {
        std::cerr << "Unsupported option: " << arg << std::endl;
        return false;
      }
      ch = it->second.first;
      if (it->second.second) {
        if (i + 1 >= argc) {
          std::cerr << "Option " << arg << " requires an argument."
                    << std::endl;
          return false;
        }
        val = argv[++i];
      }
    } else if (arg.compare(0, 1, "-") == 0 && arg.size() == 2) {
      ch = arg[1];
      bool found = false;
      for (auto &kv : opt_map) {
        if (kv.second.first == ch) {
          if (kv.second.second) {
            if (i + 1 >= argc) {
              std::cerr << "Option " << arg << " requires an argument."
                        << std::endl;
              return false;
            }
            val = argv[++i];
          }
          found = true;
          break;
        }
      }
      if (!found) {
        std::cerr << "Unsupported option: " << arg << std::endl;
        return false;
      }
    } else {
      std::cerr << "Unexpected argument: " << arg << std::endl;
      return false;
    }

    switch (ch) {
      case 'h':
        PrintUsage();
        return false;
      case 'm':
        arguments->model_path = val;
        break;
      case 'n':
        arguments->model_name = val;
        break;
      case 'i':
        arguments->input_path = val;
        break;
      case 'w':
        arguments->warm_up = atoi(val.c_str());
        break;
      case 'b':
        arguments->batch = atoi(val.c_str());
        break;
      case 't':
        arguments->threads = atoi(val.c_str());
        break;
      case 'd':
        arguments->devices = atoi(val.c_str());
        break;
      case 'l':
        arguments->loops = atoi(val.c_str());
        break;
      case 's':
        arguments->samples = atoi(val.c_str());
        break;
      case 'o':
        arguments->output_path = val;
        break;
      case 'y':
        arguments->infer_only = (val == "1" || val == "true");
        break;
      case 'p':
        arguments->module_pool = (val == "1" || val == "true");
        break;
      case 'e':
        arguments->streams = atoi(val.c_str());
        break;
      case 'c':
        arguments->modules = atoi(val.c_str());
        break;
      case 'v':
        arguments->interval = atoi(val.c_str());
        break;
      case 'q':
        arguments->queue_length = atoi(val.c_str());
        break;
    }
  }
  return true;
}

template <typename T>
std::ostream &operator<<(std::ostream &out, const std::vector<T> &vec) {
  out << "[";
  for (size_t idx = 0; idx < vec.size(); idx++) {
    if (idx != 0) {
      out << ", ";
    }
    out << vec[idx];
  }
  out << "]";
  return out;
}

#ifdef XH2A_HM_SYS
int GetDevMemInfo(std::map<int, hm_mem_info> &dev_mem_info) {
  hm_device_info dev_info = {0};
  int ret = hm_sys_get_device_info(&dev_info);
  if (ret <= 0 || dev_info.num_devices <= 0) {
    LOG_ERROR("Not found online devices, ret is {}.", ret);
    return -1;
  }

  LOG_INFO("Online device num: {}.", dev_info.num_devices);
  for (int i = 0; i < dev_info.num_devices; i++) {
    int device_id = dev_info.device_ids[i];
    dev_mem_info[device_id] = {0};
    ret = hm_sys_get_mem_info(device_id, &dev_mem_info[device_id]);
    if (ret != 0) {
      LOG_ERROR("Failed to get memory info of device {}, ret is {}.", device_id,
                ret);
      return ret;
    }
    auto mem_info = dev_mem_info[device_id];
    LOG_INFO(
        "Online device id: {}, mem_total: {}, mem_used: {}, mem_avail: {}.",
        device_id, mem_info.mem_total, mem_info.mem_used, mem_info.mem_avail);
  }

  return ret;
}
#endif

class Barrier {
 public:
  Barrier(int dest) : dest_(dest) {}

  void barrier() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_++;
    cond0_.notify_all();
    cond_.wait(lock);
  }

  bool wait(int timeout = 0) {
    std::unique_lock<std::mutex> lock(mtx_);
    int time = 0;
    while (count_ < dest_) {
      if (timeout == 0) {
        cond0_.wait(lock);
      } else {
        lock.unlock();
        time += 10;
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        lock.lock();
        if (time >= timeout) {
          return false;
        }
      }
    }
    cond_.notify_all();
    return true;
  }

  void barrier_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_++;
    if (count_ < dest_) {
      cond_.wait(lock);
    } else {
      cond_.notify_all();
      cond0_.notify_all();
    }
  }

  void reset() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_ = 0;
  }

 protected:
  std::atomic<int> count_{0};
  int dest_ = 0;
  std::condition_variable cond_, cond0_;
  std::mutex mtx_;
};

// void SetAffinity(int core_id) {
//   cpu_set_t cpuset;
//   CPU_ZERO(&cpuset);
//   CPU_SET(core_id, &cpuset);
//   // sched_setaffinity(getpid(), sizeof(cpu_set_t), &cpuset);

//   if (pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset) !=
//   0) {
//     perror("pthread_setaffinity_np");
//     exit(EXIT_FAILURE);
//   }
// }

std::string SanitizeName(const std::string &name) {
  std::string output = name;
  for (char &c : output) {
    if (c == '/' || c == '-' || c == '#') {
      c = '_';
    }
  }
  return output;
}

template <typename T>
void LoadNpyFile(const std::string &data_file, std::vector<size_t> &shape,
                 tcim::Tensor &tensor) {
  std::vector<T> data;
#ifdef _MSC_VER
  std::vector<npy::ndarray_len_t> npy_shape;
  // When calling LoadArrayFromNumpy, pass npy_shape
  // (instead of the original size_t type
  // shape)
  npy::LoadArrayFromNumpy(data_file, npy_shape, data);
  // Convert npy_shape back to size_t type shape (to meet function
  // output requirements)
  shape.clear();
  for (auto dim : npy_shape) {
    // Safe conversion (ndarray_len_t is typically int64_t/size_t)
    shape.emplace_back(static_cast<size_t>(dim));
  }
  memcpy(tensor.Data(), data.data(), tensor.Info().MemSize());
#else
  npy::LoadArrayFromNumpy(data_file, shape, data);
  memcpy(tensor.Data(), &data[0], tensor.Info().MemSize());
#endif
}

int LoadNpy2Tensor(const std::string &hm_target, const std::string &data_file,
                   tcim::DataType data_type, tcim::Tensor &tensor) {
  LOG_DEBUG("Load Numpy array file to tensor, tensor datatype: {}, file: {}.",
            static_cast<int>(data_type), data_file);
  if (!fs::exists(data_file)) {
    LOG_ERROR("File {} doesn't exist.", data_file);
    return -1;
  }

  std::string kv_cache_str = "cache";
  std::vector<size_t> shape;
  switch (data_type) {
    case 0:  // tcim::DataType::INT8
      if (data_file.find(kv_cache_str) != std::string::npos) {
        return -3;
      }
      LoadNpyFile<signed char>(data_file, shape, tensor);
      break;
    case 1:  // tcim::DataType::UINT8
      if (data_file.find(kv_cache_str) != std::string::npos) {
        return -3;
      }
      LoadNpyFile<unsigned char>(data_file, shape, tensor);
      break;
    case 2:  // tcim::DataType::INT16
      LoadNpyFile<short>(data_file, shape, tensor);
      break;
    case 3:  // tcim::DataType::UINT16
      LoadNpyFile<unsigned short>(data_file, shape, tensor);
      break;
    case 4:  // tcim::DataType::INT32
      LoadNpyFile<int>(data_file, shape, tensor);
      break;
    case 5:  // tcim::DataType::UINT32
      LoadNpyFile<unsigned int>(data_file, shape, tensor);
      break;
    case 6:  // tcim::DataType::FLOAT16
      LoadNpyFile<npy::float16>(data_file, shape, tensor);
      break;
    case 7:  // tcim::DataType::FLOAT32
      LoadNpyFile<float>(data_file, shape, tensor);
      break;
    default:
      LOG_DEBUG("Unsupported datatype {}.", static_cast<int>(data_type));
      return -2;
  }

  return 0;
}

int PrepareInputs(PooledModule *pooled_md, tcim::Module *module,
                  const std::string &hm_target, const std::string &model_name,
                  const std::string &input_path,
                  std::vector<std::string> &image_input_names,
                  std::map<std::string, tcim::Tensor> &input_datas,
                  bool &is_result_check) {
  int input_num = 0;
  std::string custom_msg_str = "";
  if (pooled_md != nullptr) {
    custom_msg_str = pooled_md->GetCustomMsg();
    input_num = pooled_md->GetInputNum();
  } else if (module != nullptr) {
    custom_msg_str = module->GetCustomMsg();
    input_num = module->GetInputNum();
  } else {
    return -1;
  }

  int ret = 0;
  LOG_INFO("Count of Input: {}", input_num);
  for (int idx = 0; idx < input_num; idx++) {
    std::string input_name = "";
    tcim::TensorInfo input_info;
    if (pooled_md != nullptr) {
      input_name = pooled_md->GetInputName(idx);
      input_info = pooled_md->GetInputInfo(input_name, false);
    } else {
      input_name = module->GetInputName(idx);
      input_info = module->GetInputInfo(input_name);
    }
    LOG_INFO("  Input[{}] {}.", input_name, TensorInfo2Str(input_info));
    input_info = input_info.AsContiguous();
    auto tensor = tcim::Tensor::CreateHostTensor(input_info);

    if (!input_path.empty()) {
      // load input data from the specified input folder
      auto data_type = input_info.DataType();
      auto data_file = input_path + "/hmquant_" + model_name + "_" +
                       SanitizeName(input_name) + "_input.npy";
      ret = LoadNpy2Tensor(hm_target, data_file, data_type, tensor);
      if (ret != 0) {
        LOG_WARNING(
            "Failed to load input data file {}. Use random data and "
            "result check will be skipped.",
            data_file);
        is_result_check = false;
      }
    }

    auto fmt = input_info.Format();
    if (fmt == tcim::DataFmt::YUV420SP || fmt == tcim::DataFmt::YUV422SP ||
        fmt == tcim::DataFmt::YUV444SP) {
      image_input_names.emplace_back(input_name);
    }
    input_datas.insert(
        std::pair<std::string, tcim::Tensor>(input_name, tensor));
  }

  // prepare dynamic inputs
  for (auto &name : image_input_names) {
    std::string dyn_name = "resizer_crop_" + name;
    if (input_datas.count(dyn_name) == 1) continue;
    if (custom_msg_str.empty()) {
      LOG_ERROR("[error] Custom message is empty.");
      return -2;
    }
    json custom_msg = json::parse(custom_msg_str);
    auto &resizer_mode = custom_msg[name]["resizer_mode"];
    if (resizer_mode != 1 && resizer_mode != 2) continue;
    std::vector<int64_t> img_shape;
    std::vector<int64_t> dyn_shape;
    if (pooled_md != nullptr) {
      img_shape = pooled_md->GetInputInfo(name, false).Shape();
      dyn_shape = pooled_md->GetInputInfo(dyn_name, false).Shape();
    } else {
      img_shape = module->GetInputInfo(name).Shape();
      dyn_shape = module->GetInputInfo(dyn_name).Shape();
    }
    auto &model_input_shape = custom_msg[name]["shape"];
    if (img_shape.size() != 4) {
      LOG_ERROR("[error] Input [{}] image shape should be 4 dims.", name);
      return -3;
    }
    if (model_input_shape.size() != 4) {
      LOG_ERROR("[error] Input [{}] model shape should be 4 dims.", name);
      return -4;
    }
    if (dyn_shape.size() != 2 && dyn_shape.size() != 1) {
      LOG_ERROR("[error] Input [{}] dyn shape should be 1 or 2 dims.", name);
      return -5;
    }

    int32_t *dyn_data =
        reinterpret_cast<int32_t *>(input_datas[dyn_name].Data());
    int32_t batch = 1;
    int32_t step = 4;
    if (dyn_shape.size() == 2) {
      batch = dyn_shape[0];
      step = dyn_shape[1];
    } else if (dyn_shape.size() == 1) {
      step = dyn_shape[0];
    }
    for (int i = 0; i < batch; ++i) {
      dyn_data[i * step + 0] = 0;
      dyn_data[i * step + 1] = 0;
      dyn_data[i * step + 2] = img_shape[2];
      dyn_data[i * step + 3] = img_shape[3];
      if (step == 10) {
        dyn_data[i * step + 4] = model_input_shape[2];
        dyn_data[i * step + 5] = model_input_shape[3];
        dyn_data[i * step + 6] = 0;
        dyn_data[i * step + 7] = 0;
        dyn_data[i * step + 8] = 0;
        dyn_data[i * step + 9] = 0;
      }
    }
  }

  return 0;
}

int PrepareOptGolden(PooledModule *pooled_md, tcim::Module *module,
                     const std::string &hm_target,
                     const std::string &model_name,
                     const std::string &input_path,
                     std::map<std::string, tcim::Tensor> &output_golden,
                     bool &is_result_check) {
  int output_num = 0;
  if (pooled_md != nullptr) {
    output_num = pooled_md->GetOutputNum();
  } else if (module != nullptr) {
    output_num = module->GetOutputNum();
  } else {
    return -1;
  }

  int ret = 0;
  LOG_INFO("Count of Output: {}", output_num);
  for (int idx = 0; idx < output_num; idx++) {
    std::string output_name = "";
    tcim::TensorInfo output_info;
    if (pooled_md != nullptr) {
      output_name = pooled_md->GetOutputName(idx);
      output_info = pooled_md->GetOutputInfo(output_name, false);
    } else {
      output_name = module->GetOutputName(idx);
      output_info = module->GetOutputInfo(output_name);
    }
    LOG_INFO("  Output[{}] {}.", output_name, TensorInfo2Str(output_info));
    auto data_type = output_info.DataType();
    output_info = output_info.AsContiguous();
    auto tensor = tcim::Tensor::CreateHostTensor(output_info);
    if (is_result_check && !input_path.empty()) {
      auto data_file = input_path + "/hmquant_" + model_name + "_" +
                       SanitizeName(output_name) + "_output.npy";
      ret = LoadNpy2Tensor(hm_target, data_file, data_type, tensor);
      if (ret != 0) {
        LOG_WARNING(
            "Failed to load output data file {}. Result check will be "
            "skipped.",
            data_file);
        is_result_check = false;
      }
    }
    output_golden.insert(
        std::pair<std::string, tcim::Tensor>(output_name, tensor));
  }

  return 0;
}

int ModulePoolFunc(ModulePool *module_pool, int tid, int did, ThreadInfo &info,
                   TaskQueue &qin, TaskQueue &qout, Barrier &barrier) {
  auto start = GET_TIME();
  auto end = GET_TIME();
  int ret = 0;
  float cost = 0.0;
  // load model
  start = GET_TIME();
  std::unique_lock<std::mutex> lock_xx(qin.mutex);
  auto option = tcim::Module::Option(info.weight_manager);
  auto pooled_md = module_pool->Load(info.model_path, option);
  lock_xx.unlock();
  end = GET_TIME();
  cost = GET_COST(start, end) / 1000.0 / info.warm_up;
  if (pooled_md == nullptr) {
    barrier.barrier();
    LOG_ERROR("Device {} Thread {} load model {} fail.", did, tid,
              info.model_path);
    return -1;
  }
  LOG_INFO("Device {} Thread {} {} model loaded. Cost {} ms.", did, tid,
           info.model_path, cost);

  {
    // warm up
    std::lock_guard<std::mutex> lock_in(qin.mutex);
    auto task0 = qin.queue.front();
    std::vector<int32_t> wm_perf_stats;
    start = GET_TIME();
    for (int i = 0; i < info.warm_up; i++) {
      ret = pooled_md->Infer(task0.data_in, task0.data_out, wm_perf_stats, 1);
    }
    end = GET_TIME();
    cost = GET_COST(start, end) / 1000.0 / info.warm_up;
    LOG_INFO("Device {} Thread {} Warm Up {} average cost {} ms.", did, tid,
             info.warm_up, cost);
  }

  int infer_stage = info.infer_only == true ? 3 : 0;
  // wait until all threads ready
  barrier.barrier();
  LOG_INFO("Device {} Thread {} infer start...", did, tid);
  int count = 0;

  while (true) {
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task = qin.queue.front();
    if (task.req_id == -1) {
      qin.cond.notify_all();
      lock_in.unlock();
      break;
    }
    qin.queue.pop();
    lock_in.unlock();

    LOG_DEBUG("Device {} Thread {} ready to infer task {}", did, tid,
              task.req_id);
    tcim::Module::RunOption run_option;
    run_option.Rounds(info.loop_num);
    std::vector<int32_t> perf_stats;
    start = GET_TIME();
    ret = pooled_md->Infer(task.data_in, task.data_out, perf_stats, infer_stage,
                           run_option);
    end = GET_TIME();
    cost = GET_COST(start, end);

    info.e2e_total_cost += cost;
    if (info.e2e_max_cost < cost) {
      // LOG_INFO(
      //     "Device {} Thread {} infer task {}, update max e2e cost {}, input
      //     "
      //     "{}, infer {}, output {}",
      //     did, tid, task.req_id, (cost / 1000.0), (perf_stats[0] / 1000.0),
      //     (perf_stats[1] / 1000.0), (perf_stats[2] / 1000.0));
      info.e2e_max_cost = cost;
    }
    if (info.e2e_min_cost > cost) info.e2e_min_cost = cost;

    info.input_total_cost += perf_stats[0];
    if (info.input_max_cost < perf_stats[0])
      info.input_max_cost = perf_stats[0];
    if (info.input_min_cost > perf_stats[0])
      info.input_min_cost = perf_stats[0];

    info.infer_total_cost += perf_stats[1];
    if (info.infer_max_cost < perf_stats[1])
      info.infer_max_cost = perf_stats[1];
    if (info.infer_min_cost > perf_stats[1])
      info.infer_min_cost = perf_stats[1];

    info.output_total_cost += perf_stats[2];
    if (info.output_max_cost < perf_stats[2])
      info.output_max_cost = perf_stats[2];
    if (info.output_min_cost > perf_stats[2])
      info.output_min_cost = perf_stats[2];

    if (!info.infer_only) {
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task);
      lock_out.unlock();
    }
    count++;
  }
  info.sample_cnt = count;
  LOG_INFO("Device {} Thread {} completed. {} samples tested.", did, tid,
           info.sample_cnt);

  delete pooled_md;

  return 0;
}

int StreamEngineFunc(int tid, int did, ThreadInfo &info, StreamEngine *engine,
                     TaskQueue &qin, TaskQueue &qout, Barrier &barrier) {
  auto start = GET_TIME();
  auto end = GET_TIME();
  float cost = 0.0;
  // load model
  start = GET_TIME();
  std::unique_lock<std::mutex> lock_xx(qin.mutex);
  auto option = tcim::Module::Option(info.weight_manager);
  auto module = tcim::Module::LoadFromFile(info.model_path, option);
  lock_xx.unlock();
  end = GET_TIME();
  cost = GET_COST(start, end) / 1000.0 / info.warm_up;
  if (module.GetInitStatus() != tcim::Status::OK) {
    barrier.barrier();
    LOG_ERROR("Device {} Thread {} load model {} fail.", did, tid,
              info.model_path);
    return -1;
  }
  LOG_INFO("Device {} Thread {} {} model loaded. Cost {} ms.", did, tid,
           info.model_path, cost);

  {
    // warm up
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    auto task0 = qin.queue.front();
    for (auto &tensor : task0.data_in) {
      module.SetInput(tensor.first, tensor.second);
    }
    lock_in.unlock();
    start = GET_TIME();
    for (int i = 0; i < info.warm_up; i++) {
      module.Run(false);
    }
    module.Sync();
    end = GET_TIME();
    cost = GET_COST(start, end) / 1000.0 / info.warm_up;
    LOG_INFO("Device {} Thread {} Warm Up {} average cost {} ms.", did, tid,
             info.warm_up, cost);
  }

  // wait until all threads ready
  barrier.barrier();
  LOG_INFO("Device {} Thread {} infer start...", did, tid);

  int count = 0;

  while (true) {
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task = qin.queue.front();
    if (task.req_id == -1) {
      lock_in.unlock();
      qin.cond.notify_all();
      break;
    }
    qin.queue.pop();
    lock_in.unlock();

    LOG_DEBUG("Device {} Thread {} ready to infer task {}", did, tid,
              task.req_id);
    start = GET_TIME();
    if (!info.infer_only) {
      for (auto &tensor : task.data_in) {
        module.SetInput(tensor.first, tensor.second);
      }
    }
    auto input_end = GET_TIME();
    auto ipt_cost = GET_COST(start, input_end);
    info.input_total_cost += ipt_cost;
    if (info.input_max_cost < ipt_cost) info.input_max_cost = ipt_cost;
    if (info.input_min_cost > ipt_cost) info.input_min_cost = ipt_cost;

    tcim::Module::RunOption run_option;
    run_option.Rounds(info.loop_num);
    engine->RunSync(module, run_option);

    auto infer_end = GET_TIME();
    auto infer_cost = GET_COST(input_end, infer_end);
    info.infer_total_cost += infer_cost;
    if (info.infer_max_cost < infer_cost) info.infer_max_cost = infer_cost;
    if (info.infer_min_cost > infer_cost) info.infer_min_cost = infer_cost;

    if (!info.infer_only) {
      for (auto &tensor : task.data_out) {
        // auto output_start = GET_TIME();
        module.GetOutput(tensor.first, tensor.second);
        // auto output_end = GET_TIME();
        // cost = GET_COST(output_start, output_end);
        // std::cout << "GetOutput " << tensor.first << ": " << cost / 1000.0
        // << " ms" << std::endl;
      }
    }
    end = GET_TIME();
    auto opt_cost = GET_COST(infer_end, end);
    info.output_total_cost += opt_cost;
    if (info.output_max_cost < opt_cost) info.output_max_cost = opt_cost;
    if (info.output_min_cost > opt_cost) info.output_min_cost = opt_cost;

    cost = GET_COST(start, end);
    info.e2e_total_cost += cost;
    if (info.e2e_max_cost < cost) {
      // LOG_INFO(
      //     "Device {} Thread {} infer task {}, update max e2e cost {}, input
      //     "
      //     "{}, infer {}, output {}",
      //     did, tid, task.req_id, (cost / 1000.0), (ipt_cost / 1000.0),
      //     (infer_cost / 1000.0), (opt_cost / 1000.0));
      info.e2e_max_cost = cost;
    }
    if (info.e2e_min_cost > cost) info.e2e_min_cost = cost;

    if (!info.infer_only) {
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task);
      lock_out.unlock();
    }
    count++;
  }
  info.sample_cnt = count;
  LOG_INFO("Device {} Thread {} completed. {} samples tested.", did, tid,
           info.sample_cnt);

  return 0;
}

void MainThreadWarm(PooledModule *pooled_md, tcim::Module *module, int warm_up,
                    std::map<std::string, tcim::Tensor> &input_datas,
                    std::map<std::string, tcim::Tensor> &output_golden) {
  auto start = GET_TIME();
  auto end = GET_TIME();

  // create tmp outputs
  std::map<std::string, tcim::Tensor> output_datas;
  for (auto &output : output_golden) {
    auto info = output.second.Info().AsContiguous();
    auto tensor = tcim::Tensor::CreateHostTensor(info);
    output_datas.insert(
        std::pair<std::string, tcim::Tensor>(output.first, tensor));
  }

  if (pooled_md != nullptr) {
    std::vector<int32_t> wm_perf_stats;
    start = GET_TIME();
    for (int i = 0; i < warm_up; i++) {
      auto ret = pooled_md->Infer(input_datas, output_datas, wm_perf_stats, 1);
    }
    end = GET_TIME();
  } else {
    for (auto &tensor : input_datas) {
      module->SetInput(tensor.first, tensor.second);
    }
    start = GET_TIME();
    for (int i = 0; i < warm_up; i++) {
      module->Run(false);
    }
    module->Sync();
    end = GET_TIME();
  }
  auto cost = GET_COST(start, end) / 1000.0 / warm_up;
  LOG_INFO("Main Thread Warm Up {} average cost {} ms.", warm_up, cost);
}

void ProcessPerfResults(
    const ThreadInfo *thread_info, size_t thread_num, int loop_num,
    int sample_num, int batch, float total_cost, const std::string &output_path,
    const std::map<std::string, tcim::Tensor> &input_datas) {
  uint32_t infer_min_cost = 999999;
  uint32_t infer_max_cost = 0;
  uint32_t infer_total_cost = 0;
  uint32_t input_min_cost = 999999;
  uint32_t input_max_cost = 0;
  uint32_t input_total_cost = 0;
  uint32_t output_min_cost = 999999;
  uint32_t output_max_cost = 0;
  uint32_t output_total_cost = 0;
  uint32_t e2e_min_cost = 999999;
  uint32_t e2e_max_cost = 0;
  uint32_t e2e_total_cost = 0;

  for (int i = 0; i < thread_num; i++) {
    if (thread_info[i].infer_min_cost < infer_min_cost) {
      infer_min_cost = thread_info[i].infer_min_cost;
    }
    if (thread_info[i].infer_max_cost > infer_max_cost) {
      infer_max_cost = thread_info[i].infer_max_cost;
    }
    infer_total_cost += thread_info[i].infer_total_cost;

    if (thread_info[i].input_min_cost < input_min_cost) {
      input_min_cost = thread_info[i].input_min_cost;
    }
    if (thread_info[i].input_max_cost > input_max_cost) {
      input_max_cost = thread_info[i].input_max_cost;
    }
    input_total_cost += thread_info[i].input_total_cost;

    if (thread_info[i].output_min_cost < output_min_cost) {
      output_min_cost = thread_info[i].output_min_cost;
    }
    if (thread_info[i].output_max_cost > output_max_cost) {
      output_max_cost = thread_info[i].output_max_cost;
    }
    output_total_cost += thread_info[i].output_total_cost;

    if (thread_info[i].e2e_min_cost < e2e_min_cost) {
      e2e_min_cost = thread_info[i].e2e_min_cost;
    }
    if (thread_info[i].e2e_max_cost > e2e_max_cost) {
      e2e_max_cost = thread_info[i].e2e_max_cost;
    }
    e2e_total_cost += thread_info[i].e2e_total_cost;
  }

  int test_num = loop_num * sample_num;  // only loop inference
  float infer_avg_latency = infer_total_cost / test_num / 1000.0;
  float infer_max_latency = infer_max_cost / 1000.0;
  float infer_min_latency = infer_min_cost / 1000.0;
  float input_avg_latency = input_total_cost / sample_num / 1000.0;
  float input_max_latency = input_max_cost / 1000.0;
  float input_min_latency = input_min_cost / 1000.0;
  float output_avg_latency = output_total_cost / sample_num / 1000.0;
  float output_max_latency = output_max_cost / 1000.0;
  float output_min_latency = output_min_cost / 1000.0;
  float e2e_avg_latency = e2e_total_cost / test_num / 1000.0;
  float e2e_max_latency = e2e_max_cost / 1000.0;
  float e2e_min_latency = e2e_min_cost / 1000.0;
  float avg_cost = total_cost / test_num;
  float qps = (1000.0 / (total_cost / test_num)) * batch;

  LOG_INFO(
      "[latency] {:8} \tavg: {:8.3f} ms, \tmax: {:8.3f} ms, \tmin: "
      "{:8.3f} ms",
      "Inference", infer_avg_latency, infer_max_latency, infer_min_latency);
  LOG_INFO(
      "[latency] {:8} \tavg: {:8.3f} ms, \tmax: {:8.3f} ms, \tmin: {:8.3f} "
      "ms",
      "Input", input_avg_latency, input_max_latency, input_min_latency);
  LOG_INFO(
      "[latency] {:8} \tavg: {:8.3f} ms, \tmax: {:8.3f} ms, \tmin: {:8.3f} "
      "ms",
      "Output", output_avg_latency, output_max_latency, output_min_latency);
  LOG_INFO(
      "[latency] {:8} \tavg: {:8.3f} ms, \tmax: {:8.3f} ms, \tmin: {:8.3f} "
      "ms",
      "End2End", e2e_avg_latency, e2e_max_latency, e2e_min_latency);
  LOG_INFO("[Throughput] total: {:8.3f} ms, avg: {:8.3f} ms", total_cost,
           avg_cost);
  LOG_INFO("[Throughput] qps: {:8.3f}", qps);

  if (!output_path.empty()) {
    std::string output_file = output_path + "/hmperf.txt";
    LOG_INFO("Save result to: {}", output_file);
    std::fstream result_file(output_file.c_str(), std::ios::out);
    result_file << "batch: " << batch << std::endl;
    result_file << "thread_num: " << thread_num << std::endl;
    result_file << "shape: [";
    for (auto &input : input_datas) {
      result_file << input.second.Info().Shape() << ",";
    }
    result_file << "]" << std::endl;
    result_file << "loop_num: " << loop_num << std::endl;
    result_file << "sample_num: " << sample_num << std::endl;
    result_file << "avg_latency: " << infer_avg_latency << std::endl;
    result_file << "max_latency: " << infer_max_latency << std::endl;
    result_file << "min_latency: " << infer_min_latency << std::endl;
    result_file << "qps: " << qps << std::endl;
    result_file.close();
  }
}

int PerfFunc(CliArguments &arguments) {
  // Load parameters
  std::string model_path = arguments.model_path;
  std::string input_path = arguments.input_path;
  std::string output_path = arguments.output_path;
  std::string model_name = arguments.model_name;
  int batch = arguments.batch;
  int sample_num = arguments.samples;
  int thread_num = arguments.threads;
  int device_num = arguments.devices;
  int loop_num = arguments.loops;
  int warm_up = arguments.warm_up;
  bool infer_only = arguments.infer_only;
  bool use_md_pool = arguments.module_pool;
  int stream_num = arguments.streams;
  int module_num = arguments.modules;
  int interval = arguments.interval > 0 ? arguments.interval : 0;
  int queue_length = arguments.queue_length;
  std::string hm_target = "";
  bool is_result_check = true;
  int device_id = 0;
  auto start = GET_TIME();
  auto end = GET_TIME();
  auto cost = GET_COST(start, end);
  int ret = 0;

  if (infer_only || input_path.empty()) {
    is_result_check = false;
  }

  if (auto target = std::getenv("HOUMO_TARGET")) {
    if (!strcmp(target, "xh2")) {
      is_result_check = false;
      hm_target = "xh2";
      LOG_WARNING("xh2 not support result check. disabled.");
    } else if (!strcmp(target, "xh1")) {
      hm_target = "xh1";
    }
  }

  if (module_num == 0) {
    auto core_num = std::getenv("HOUMO_CORE_NUM");
    if (core_num) {
      module_num = (atoi(core_num) * 2);
    } else {
      module_num = 1;
    }
  }

  if (auto platform = std::getenv("HDPL_PLATFORM")) {
    if (!strcmp(platform, "ISIM")) {
      thread_num = 1;
      sample_num = 1;
      stream_num = 1;
      module_num = 1;
      LOG_WARNING("threads and samples set to 1 while HDPL_PLATFORM=ISIM.");
    }
  }

  if (auto device = std::getenv("HOUMO_DEVICES")) {
    if (device) device_id = atoi(device);
  }

  LOG_INFO(
      "\n***** TCIM PERF Params *****\n  model: {} \n  name: {}\n  input: "
      "{}\n  samples: {}\n  loops: {}\n  warmup: {}\n  batch: {}\n  threads: "
      "{}\n  device_num: {}\n  devices: {}\n  infer_only: {}\n  use module "
      "pool: {}\n  stream_num: {}\n  module_num: {}\n  interval: {}\n  queue "
      "length: {}",
      model_path, model_name, input_path, sample_num, loop_num, warm_up, batch,
      thread_num, device_num, device_id, infer_only, use_md_pool, stream_num,
      module_num, interval, queue_length);

  if (sample_num < thread_num) {
    LOG_WARNING("the perf result may not be accurate while samples threads.");
  }

  TaskQueue qin;
  TaskQueue qout;

  ModulePool *module_pool = nullptr;
  if (use_md_pool) {
    module_pool = ModulePool::Init(module_num, stream_num);
  }

  // create weight manager
  tcim::Module::WeightManager weight_manager;
  if (device_num > 1) {
    LOG_INFO("create device manager. device num = {}.", device_num);
    std::vector<int> device_vec;
    for (int i = 0; i < device_num; i++) {
      device_vec.push_back(i);
    }
    tcim::DevManager dev_manager = tcim::DevManager::Create(device_vec, "xh2");
    weight_manager =
        tcim::Module::WeightManager::CreateWeightManager(dev_manager);
  } else {
    LOG_INFO("create weight manager. device num = {}.", device_num);
    weight_manager =
        tcim::Module::WeightManager::CreateWeightManager(device_id);
  }

  // Load module from file
  auto option = tcim::Module::Option(weight_manager);
  tcim::Module *module = nullptr;
  PooledModule *pooled_md = nullptr;
#ifdef XH2A_HM_SYS
  std::map<int, hm_mem_info> dev_mem_info_start;
  auto mem_ret_start = GetDevMemInfo(dev_mem_info_start);
#endif
  if (use_md_pool) {
    pooled_md = module_pool->Load(model_path, option);
    if (pooled_md == nullptr) {
      LOG_ERROR("[error] load model {} failed, exit!", model_path);
      delete module_pool;
      return -1;
    }
  } else {
    module = new tcim::Module();
    ret = module->LoadModel(model_path, option);
    if (ret != tcim::Status::OK) {
      LOG_ERROR("[error] load model {} failed, exit!", model_path);
      delete module;
      return -1;
    }
  }
#ifdef XH2A_HM_SYS
  std::map<int, hm_mem_info> dev_mem_info_end;
  auto mem_ret_end = GetDevMemInfo(dev_mem_info_end);
  if (mem_ret_start == 0 && mem_ret_end == 0) {
    LOG_INFO("****** HM Device Memory Usage ******");
    for (const auto &pair : dev_mem_info_start) {
      int device_id = pair.first;
      const hm_mem_info &mem_info_start = pair.second;
      if (dev_mem_info_end.count(device_id) == 0) {
        LOG_WARNING("Failed to get device {} memory info.", device_id);
        break;
      }
      const hm_mem_info &mem_info_end = dev_mem_info_end[device_id];
      int32_t mem_used = mem_info_end.mem_used - mem_info_start.mem_used;
      mem_used = mem_used < 0 ? 0 : mem_used;
      LOG_INFO("Device id: {}, memory used: {} MB", device_id, mem_used);
    }
    LOG_INFO("************************************");
  } else {
    LOG_WARNING(
        "Failed to get device memory info, start ret is {}, end ret is {}.",
        mem_ret_start, mem_ret_end);
  }
#endif

  std::vector<std::string> image_input_names;
  std::map<std::string, tcim::Tensor> input_datas;
  ret = PrepareInputs(pooled_md, module, hm_target, model_name, input_path,
                      image_input_names, input_datas, is_result_check);
  if (ret != 0) {
    LOG_ERROR("[error] Failed to prepare inputs, exit!");
    delete pooled_md;
    delete module;
    delete module_pool;
    return -2;
  }

  std::map<std::string, tcim::Tensor> output_golden;
  ret = PrepareOptGolden(pooled_md, module, hm_target, model_name, input_path,
                         output_golden, is_result_check);
  if (ret != 0) {
    LOG_ERROR("[error] Failed to prepare output golden, exit!");
    delete pooled_md;
    delete module;
    delete module_pool;
    return -3;
  }

  MainThreadWarm(pooled_md, module, warm_up, input_datas, output_golden);
  if (module != nullptr) {
    delete module;
  }

  std::vector<std::map<std::string, tcim::Tensor>> output_datas;
  // Push the warm task of sub threads to the input task queue
  Task warm_task;
  warm_task.req_id = 0;
  warm_task.data_in = input_datas;
  warm_task.ref_out = output_golden;
  if (!infer_only) {
    std::map<std::string, tcim::Tensor> output_data;
    for (auto &output : output_golden) {
      auto info = output.second.Info().AsContiguous();
      auto tensor = tcim::Tensor::CreateHostTensor(info);
      output_data.insert(
          std::pair<std::string, tcim::Tensor>(output.first, tensor));
    }
    warm_task.data_out = output_data;
    output_datas.emplace_back(output_data);
  }
  qin.queue.push(warm_task);

  // create inference threads
  StreamEngine *engine = nullptr;
  std::vector<std::future<int>> threads;
  Barrier barrier(thread_num);
#ifdef _MSC_VER
  std::vector<ThreadInfo> thread_info(thread_num);
#else
  ThreadInfo thread_info[thread_num];
#endif
  int did = 0;
  if (!use_md_pool) {
    engine = new StreamEngine(stream_num);
  }
  for (int tid = 0; tid < thread_num; tid++) {
    ThreadInfo *info = &thread_info[tid];
    info->model_path = model_path;
    info->weight_manager = weight_manager;
    info->loop_num = loop_num;
    info->infer_only = infer_only;
    info->is_result_check = is_result_check;
    info->warm_up = warm_up;
    if (use_md_pool) {
      threads.emplace_back(std::async(
          std::launch::async, ModulePoolFunc, module_pool, tid, did,
          std::ref(*info), std::ref(qin), std::ref(qout), std::ref(barrier)));
    } else {
      threads.emplace_back(std::async(
          std::launch::async, StreamEngineFunc, tid, did, std::ref(*info),
          engine, std::ref(qin), std::ref(qout), std::ref(barrier)));
    }
  }

  if (interval > 0) {
    barrier.wait();
    barrier.reset();
    LOG_INFO("Interval Mode, push a task every {} milliseconds.", interval);
    start = GET_TIME();
  }

  // the task index should start from 1
  for (int idx = 1; idx < sample_num; idx++) {
    Task task;
    task.req_id = idx;
    task.data_in = input_datas;
    task.ref_out = output_golden;
    if (!infer_only) {
      std::map<std::string, tcim::Tensor> output_data;
      for (auto &output : output_golden) {
        auto info = output.second.Info().AsContiguous();
        auto tensor = tcim::Tensor::CreateHostTensor(info);
        output_data.insert(
            std::pair<std::string, tcim::Tensor>(output.first, tensor));
      }
      task.data_out = output_data;
      output_datas.emplace_back(output_data);
    }

    {
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      if (interval > 0 && queue_length > 0 && qin.queue.size() > queue_length) {
        LOG_ERROR("There are more than {} tasks queued up, stop perf test.",
                  qin.queue.size());
        break;
      }
      LOG_DEBUG("Main thread push task {}.", task.req_id);
      qin.queue.push(task);
      qin.cond.notify_one();
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(interval));
  }
  // push end task
  Task end_task;
  end_task.req_id = -1;
  {
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    qin.queue.push(end_task);
    qin.cond.notify_all();
  }

  if (interval == 0) {
    LOG_INFO("Sample queue size is: {}", (qin.queue.size() - 1));
    barrier.wait();
    barrier.reset();
    start = GET_TIME();
  }

  // wait all threads done
  for (int i = 0; i < threads.size(); i++) {
    int tmp_res = threads[i].get();
    if (tmp_res != 0) {
      LOG_ERROR("Execute thread {} failed.", i);
    }
  }
  end = GET_TIME();
  LOG_INFO("All Perf threads done.");

  // check result
  if (is_result_check) {
    bool result = true;
    while (!qout.queue.empty()) {
      auto task = qout.queue.front();
      qout.queue.pop();

      for (auto &output : task.data_out) {
        auto data1 = (char *)output.second.Data();
        auto data2 = (char *)task.ref_out.at(output.first).Data();
        int len = output.second.Info().MemSize();
        if (memcmp(data1, data2, len)) {
          int err = 0;
          for (int i = 0; i < len; i++) {
            if (data1[i] != data2[i]) {
              err++;
            }
          }
          LOG_ERROR("[error] req: {}, output: {} result check failed ({}/{})",
                    task.req_id, output.first, (len - err), len);
          result = false;
        }
      }
    }
    if (result) {
      LOG_INFO("Result check passed.");
    }
  } else {
    LOG_WARNING("Result check skipped.");
  }

  if (auto platform = std::getenv("HDPL_PLATFORM")) {
    if (!strcmp(platform, "ISIM")) {
      LOG_WARNING(
          "The performance results are simulated while "
          "HDPL_PLATFORM=ISIM.");
    }
  }

  float total_cost = GET_COST(start, end) / 1000.0;
#ifdef _MSC_VER
  ProcessPerfResults(thread_info.data(), thread_num, loop_num, sample_num,
                     batch, total_cost, output_path, input_datas);
#else
  ProcessPerfResults(thread_info, thread_num, loop_num, sample_num, batch,
                     total_cost, output_path, input_datas);
#endif

  delete engine;
  delete module_pool;

  return 0;
}

int main(int argc, char *argv[]) {
  CliArguments arguments;
  if (!ParseArgs(&arguments, argc, argv)) {
    return 0;
  }

  return PerfFunc(arguments);
}
