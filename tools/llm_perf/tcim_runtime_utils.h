
/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: tcim_runtime_utils.h
 * Description:
 *   Check TCIM Runtime API ret Code and Debug Model Input Datas.
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
#ifndef __TCIM_RUNTIME_UTILS_H__
#define __TCIM_RUNTIME_UTILS_H__

#include "tcim/tcim_runtime.h"
inline void CheckTcimRetStatus(const tcim::Status &status,
                               const char *file = __FILE__,
                               int line = __LINE__) {
  if (status != tcim::Status::OK) {
    std::ostringstream err_msg;
    err_msg << "tcim_runtime ret Status is not OK! "
            << "File: " << file << ", Line: " << line
            << ", Current ret Status: " << static_cast<int>(status);

    throw std::runtime_error(err_msg.str());
  }
}

#define CHECK_TCIM_RET_STATUS(status) \
  CheckTcimRetStatus(status, __FILE__, __LINE__)

inline void DebugSetInputValue(std::shared_ptr<tcim::Module> module,
                               int start_idx, int end_idx) {
#ifdef DEBUG_DEV_INPUT
  assert(start_idx < end_idx);
  if (module == nullptr) {
    return;
  }
  for (int idx = start_idx; idx < end_idx; ++idx) {
    auto input_name = module->GetInputName(idx);
    auto input_info = module->GetInputInfo(input_name).AsContiguous();
    auto data_type = input_info.DataType();
    auto dev_tensor = module->GetDevInput(input_name);
    if (data_type == tcim::DataType::FLOAT16) {
      dev_tensor = dev_tensor.AsType(tcim::DataType::FLOAT32);
    }
    tcim::Tensor host_tensor = dev_tensor.ToHost();
    size_t memSize = input_info.MemSize();
    std::cout << "Input[" << input_name << "] , value : ";
    if (data_type == tcim::DataType::INT8) {
      int8_t *host_data = static_cast<int8_t *>(host_tensor.Buffer().Data());
      for (int id = 0; id < memSize / sizeof(int8_t); ++id) {
        std::cout << host_data[id] << " ";
      }
      std::cout << std::endl;
    } else if (data_type == tcim::DataType::UINT8) {
      uint8_t *host_data = static_cast<uint8_t *>(host_tensor.Buffer().Data());
      for (int id = 0; id < memSize / sizeof(uint8_t); ++id) {
        std::cout << host_data[id] << " ";
      }
      std::cout << std::endl;
    } else if (data_type == tcim::DataType::INT16) {
      int16_t *host_data = static_cast<int16_t *>(host_tensor.Buffer().Data());
      for (int id = 0; id < memSize / sizeof(int16_t); ++id) {
        std::cout << host_data[id] << " ";
      }
      std::cout << std::endl;
    } else if (data_type == tcim::DataType::UINT16) {
      uint16_t *host_data =
          static_cast<uint16_t *>(host_tensor.Buffer().Data());
      for (int id = 0; id < memSize / sizeof(uint16_t); ++id) {
        std::cout << host_data[id] << " ";
      }
      std::cout << std::endl;
    } else if (data_type == tcim::DataType::INT32) {
      int32_t *host_data = static_cast<int32_t *>(host_tensor.Buffer().Data());
      for (int id = 0; id < memSize / sizeof(int32_t); ++id) {
        std::cout << host_data[id] << " ";
      }
      std::cout << std::endl;
    } else if (data_type == tcim::DataType::UINT32) {
      uint32_t *host_data =
          static_cast<uint32_t *>(host_tensor.Buffer().Data());
      for (int id = 0; id < memSize / sizeof(uint32_t); ++id) {
        std::cout << host_data[id] << " ";
      }
      std::cout << std::endl;
    } else if ((data_type == tcim::DataType::FLOAT16) ||
               (data_type == tcim::DataType::FLOAT32)) {
      float *host_data = static_cast<float *>(host_tensor.Buffer().Data());
      for (int id = 0; id < memSize / sizeof(float); ++id) {
        std::cout << host_data[id] << " ";
      }
      std::cout << std::endl;
    } else {
      std::cerr << "Unsupport DataType!" << std::endl;
    }
  }
#endif
  return;
}

#endif  // __TCIM_RUNTIME_UTILS_H__