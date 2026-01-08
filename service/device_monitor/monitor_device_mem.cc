/*
 * Copyright 2025 HOUMO AI
 *
 * File: monitor_device_mem.cc
 * Description:
 *   Monitor device memory usage by querying system information.
 *   This program retrieves device memory statistics and outputs them in a
 *   formatted way.
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

#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
#include "hm_sys.h"
#ifdef __cplusplus
}
#endif

int main(int argc, char** argv) {
  int device_id = -1;
  std::string output_file;

  int opt;
  while ((opt = getopt(argc, argv, "hd:o:")) != -1) {
    switch (opt) {
      case 'h':
        std::cout << "Usage: " << argv[0] << " [options]\n"
                  << "  -h    Show this help message\n"
                  << "  -d    (Optional) Device ID to monitor, defaults to "
                     "monitoring first device\n";
        return 0;

      case 'd':
        device_id = std::stoi(optarg);
        break;

      case '?':  // Unknown option or missing argument
        // optopt stores the unknown option
        std::cerr << "Error: Unknown option '" << char(optopt)
                  << "' or missing argument\n";
        return 1;

      default:
        return 1;
    }
  }

  hm_device_info dev_info = {0};
  int ret = hm_sys_get_device_info(&dev_info);

  if (ret <= 0 || dev_info.num_devices <= 0) {
    std::cerr << "Not found online devices." << std::endl;
    return -1;
  }

  // Output number of online devices and their IDs
  std::cout << "Online device num: " << dev_info.num_devices
            << ", online deivce id: ";
  for (int i = 0; i < dev_info.num_devices; i++) {
    std::cout << dev_info.device_ids[i] << " ";
  }
  std::cout << std::endl;

  // Validate specified device ID if provided
  if (device_id >= 0) {
    bool inCArr = std::find(std::begin(dev_info.device_ids),
                            std::end(dev_info.device_ids),
                            device_id) != std::end(dev_info.device_ids);
    if (!inCArr) {
      std::cerr << "Invalid device id " << device_id << std::endl;
      return -1;
    }
  }

  // If no device ID specified, use first available device
  if (device_id < 0) {
    device_id = dev_info.device_ids[0];
  }

  // Initialize memory info structure
  hm_mem_info mem_info = {0};
  auto now = std::chrono::system_clock::now();
  std::time_t current_time = std::chrono::system_clock::to_time_t(now);
  std::tm* local_time = std::localtime(&current_time);
  // Get memory information for the specified device
  ret = hm_sys_get_mem_info(device_id, &mem_info);
  // Output memory information with timestamp
  std::cout << "device_id: " << device_id
            << ", time: " << std::put_time(local_time, "%Y-%m-%d %H:%M:%S")
            << ", mem_total: " << mem_info.mem_total
            << ", mem_used: " << mem_info.mem_used
            << ", mem_avail: " << mem_info.mem_avail << std::endl;

  return 0;
}