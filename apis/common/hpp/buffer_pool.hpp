/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: buffer_pool.hpp
 * Description:
 *   Memory buffer pool implementation for efficient memory management.
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

#pragma once

#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <unordered_map>
#include <vector>

#include "tcim/tcim_runtime.h"

#define BUFFER_POOL_MAX_SIZE 300 * 1024 * 1024  // 300MB
#define ALIGN4K(size) (((size) + 4095) & ~4095)

/**
 * @brief Memory type enumeration
 */
typedef enum {
  DRM = 0,
  HOST,
  RESERVED,
} memType_t;

/**
 * @brief Configuration structure for buffer pool
 */
typedef struct BufferPoolCfg {
  size_t size;  // Block size, must be 4K aligned, and should not be repeated:
                // 4k/8k/16k/32k/64k/128k/256k/512k/1M/2M/4M/8M
  int32_t num;  // Number of blocks
} bufferPoolCfg_t;

/**
 * @brief Memory usage statistics structure
 */
typedef struct MemoryUsage {
  size_t block_size;         // Size of block
  int32_t total_blocks;      // Total number of blocks
  int32_t allocated_blocks;  // Number of allocated blocks
  int32_t free_blocks;       // Number of free blocks
} memoryUsage_t;

/**
 * @brief Fixed-size memory pool class for fixed-size memory blocks
 */
class BufferPoolImpl {
 public:
  ~BufferPoolImpl() = default;
  // Delete copy constructor and assignment operator
  BufferPoolImpl(const BufferPoolImpl &) = delete;
  BufferPoolImpl &operator=(const BufferPoolImpl &) = delete;

  /**
   * @brief Constructor
   * @param cfg Buffer pool configuration
   * @param mem_type Type of memory
   * @param device_id Device ID (default 0)
   */
  BufferPoolImpl(BufferPoolCfg &cfg, memType_t mem_type,
                 int32_t device_id = 0) {
    block_size_ = ALIGN4K(cfg.size);
    block_num_ = cfg.num;
    total_size_ = static_cast<int64_t>(block_size_) * block_num_;

    if (total_size_ > BUFFER_POOL_MAX_SIZE) {
      throw std::runtime_error(
          "[BufferPool] BufferPoolImpl: total_size_ > BUFFER_POOL_MAX_SIZE");
    }

    device_id_ = device_id;
    if ((mem_type == DRM || mem_type == RESERVED) &&
        (device_id < 0 || device_id >= tcim::GetDeviceNum())) {
      throw std::runtime_error(
          "[BufferPool] BufferPoolImpl: Invalid device_id");
    }
    switch (mem_type) {
      case DRM:  // DRM
        buf_ =
            tcim::Buffer::CreateDeviceBuffer(total_size_, device_id_, "", "");
        break;
      case HOST:  // Host memory
        buf_ = tcim::Buffer::CreateHostBuffer(total_size_);
        break;
      case RESERVED:  // Reserved memory
        buf_ = tcim::Buffer::CreateDeviceBuffer(total_size_, device_id_, "",
                                                "reserved");
        break;
      default:
        buf_ = tcim::Buffer();
        throw std::runtime_error("[BufferPool] Invalid memType");
    }
    // Initialize an empty queue with full memory pool
    for (int i = 0; i < block_num_; ++i) {
      auto block_buf = buf_.GetSubBuffer(block_size_, i * block_size_);
      free_bufs_.push(block_buf);
      block_free_status_[ptr_to_string(block_buf.Data())] = true;
    }
  }

  /**
   * @brief Request a memory block
   * @param size Size of memory to request, default 0 means requesting a full
   * block
   * @param timeout Timeout for requesting memory, default 0 means wait
   * indefinitely
   * @return tcim::Buffer object
   */
  tcim::Buffer Get(size_t size = 0, int32_t timeout = 0) {
    int blocks_needed = size <= 0 ? 1 : (size + block_size_ - 1) / block_size_;
    if (blocks_needed > block_num_) {
      printf("[BufferPool]: request size %ld is too large\n", size);
      return tcim::Buffer();  // Request too large
    }

    tcim::Buffer buffer;
    float wait_time = 0;  // ms
    do {
      // Get from free queue, wait for timeout if empty
      rw_mtx_.lock();
      if (!free_bufs_.empty()) {
        buffer = free_bufs_.front();
        free_bufs_.pop();
        block_free_status_[ptr_to_string(buffer.Data())] = false;
        allocated_blocks_++;
        rw_mtx_.unlock();
        return buffer;
      }
      rw_mtx_.unlock();
      if (buffer.Size() == 0) {
        wait_time += 1;
        usleep(1000);  // Delay 1ms
      }
    } while (wait_time < timeout || timeout == 0);
    printf("[BufferPool] Not found free block yet\n");
    return buffer;  // Insufficient space
  }

  /**
   * @brief Release a buffer
   * @param buffer Buffer to be released
   */
  void Free(const tcim::Buffer &buffer) {
    if (buffer.Size() == 0) {
      return;
    }
    std::string key = ptr_to_string(buffer.Data());
    rw_mtx_.lock();
    // External created buffer, ignore
    if (block_free_status_.find(key) == block_free_status_.end()) {
      rw_mtx_.unlock();
      printf("[BufferPool] Buffer is external\n");
      return;
    }
    // Already freed, return directly for duplicate free
    if (block_free_status_[key]) {
      rw_mtx_.unlock();
      printf("[BufferPool] Buffer has been freed\n");
      return;
    }
    block_free_status_[key] = true;
    free_bufs_.push(buffer);
    allocated_blocks_--;  // Decrease counter when releasing
    rw_mtx_.unlock();
  }

  /**
   * @brief Get memory usage statistics
   * @return MemoryUsage object with usage statistics
   */
  MemoryUsage GetUsage() {
    MemoryUsage usage;
    usage.block_size = block_size_;
    usage.total_blocks = block_num_;
    rw_mtx_.lock();
    usage.allocated_blocks = allocated_blocks_;
    usage.free_blocks = block_num_ - allocated_blocks_;
    rw_mtx_.unlock();
    return usage;
  }

 private:
  template <typename T>
  std::string ptr_to_string(T *p) {
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%p", (void *)p);
    return std::string(buf);
  }

 private:
  int32_t allocated_blocks_ = 0;  // Added statistics variable
  int32_t block_size_ = 0;
  int32_t block_num_ = 0;
  int32_t device_id_ = 0;
  int64_t total_size_ = 0;
  tcim::Buffer buf_;   // Total memory
  std::mutex rw_mtx_;  // Memory operation mutex
  std::unordered_map<std::string, bool> block_free_status_;
  std::queue<tcim::Buffer> free_bufs_;
};

/**
 * @brief Multi-size block memory pool
 */
class BufferPool {
 public:
  // Delete copy constructor and assignment operator
  BufferPool(const BufferPool &) = delete;
  BufferPool &operator=(const BufferPool &) = delete;

  /**
   * @brief Constructor
   * @param cfgs Vector of memory block configurations
   * @param mem_type Type of memory
   * @param device_id Device ID (default 0)
   */
  BufferPool(std::vector<BufferPoolCfg> &cfgs, memType_t mem_type,
             int32_t device_id = 0) {
    // Parameter validation
    if (cfgs.empty()) {
      throw std::runtime_error("[BufferPool] BufferPool: cfgs is empty");
    }
    if (mem_type != DRM && mem_type != HOST && mem_type != RESERVED) {
      throw std::runtime_error("[BufferPool] Invalid memType");
    }
    for (auto &cfg : cfgs) {
      if (cfg.size <= 0) {
        throw std::runtime_error(
            "[BufferPool] BufferPoolCfg size must be greater than 0");
      }
    }
    // Sort in ascending order
    std::sort(cfgs.begin(), cfgs.end(),
              [](const BufferPoolCfg &a, const BufferPoolCfg &b) {
                return ALIGN4K(a.size) < ALIGN4K(b.size);
              });
    for (int i = 0; i < cfgs.size() - 1; ++i) {
      auto &cfg0 = cfgs[i + 0];
      auto &cfg1 = cfgs[i + 1];
      if (cfg0.size % 4096 != 0 || cfg1.size % 4096 != 0) {
        throw std::runtime_error(
            "[BufferPool] BufferPoolCfg size must be aligned to 4096");
      }
      if (cfg0.size == cfg1.size) {
        throw std::runtime_error(
            "[BufferPool] BufferPoolCfg size must be not equal");
      }
    }
    buffers_.clear();
    block_sizes_.clear();
    for (auto &cfg : cfgs) {
      size_t block_size = ALIGN4K(cfg.size);
      std::unique_ptr<BufferPoolImpl> pool =
          std::make_unique<BufferPoolImpl>(cfg, mem_type);
      block_sizes_.emplace_back(block_size);
      buffers_[block_size] =
          std::move(pool);  // Create mapping between block_size and id
    }
  }

  /**
   * @brief Request a buffer
   * @param size Size of buffer to request
   * @param timeout Timeout for requesting memory, default 0 means wait
   * indefinitely
   * @return tcim::Buffer object
   */
  tcim::Buffer Malloc(size_t size, int timeout = 0) {
    auto it = std::lower_bound(block_sizes_.begin(), block_sizes_.end(),
                               ALIGN4K(size));
    if (it == block_sizes_.end()) {
      printf("[BufferPool] No suitable block size found, and expect size: %d\n",
             size);
      return tcim::Buffer();
    }
    int32_t block_size = *it;
    auto &pool = buffers_[block_size];
    return pool->Get(0, timeout);
  }

  /**
   * @brief Release a buffer
   * @param buffer Buffer to be released
   */
  void Free(tcim::Buffer &buffer) {
    if (buffer.Size() == 0) return;
    int32_t block_size = buffer.Size();
    auto &pool = buffers_[block_size];
    pool->Free(buffer);
  }

  /**
   * @brief Get buffer pool usage statistics
   * @return Map of block size to memory usage statistics
   */
  std::map<size_t, MemoryUsage> GetStats() {
    std::map<size_t, MemoryUsage> stats;
    for (size_t block_size : block_sizes_) {
      stats[block_size] = buffers_[block_size]->GetUsage();
    }
    return stats;
  }

 private:
  std::vector<size_t> block_sizes_;
  std::unordered_map<int32_t, std::unique_ptr<BufferPoolImpl>> buffers_;

  static inline std::unique_ptr<BufferPool> instance_;
  static inline std::once_flag init_flag_;
};

/**
 * @brief Singleton multi-size block memory pool
 */
class BufferPoolSingleton {
 public:
  // Delete copy constructor and assignment operator
  BufferPoolSingleton(const BufferPoolSingleton &) = delete;
  BufferPoolSingleton &operator=(const BufferPoolSingleton &) = delete;

  // Must be called once to initialize the singleton (thread-safe)
  static void Init(std::vector<BufferPoolCfg> &cfgs, memType_t mem_type,
                   int32_t device_id = 0) {
    std::call_once(init_flag_, [&]() {
      instance_.reset(new BufferPoolSingleton(cfgs, mem_type, device_id));
    });
  }

  // Get singleton reference (throws if not initialized)
  static BufferPoolSingleton &GetInstance() {
    if (!instance_) {
      throw std::runtime_error("[BufferPool] GetInstance called before Init");
    }
    return *instance_;
  }

  static bool IsInitialized() { return instance_ != nullptr; }

  /**
   * @brief Request a buffer
   * @param size Size of buffer to request
   * @param timeout Timeout for requesting memory, default 0 means wait
   * indefinitely
   * @return tcim::Buffer object
   */
  tcim::Buffer Malloc(size_t size, int timeout = 0) {
    auto it = std::lower_bound(block_sizes_.begin(), block_sizes_.end(),
                               ALIGN4K(size));
    if (it == block_sizes_.end()) {
      printf("[BufferPool] No suitable block size found, and expect size: %d\n",
             size);
      return tcim::Buffer();
    }
    int32_t block_size = *it;
    auto &pool = buffers_[block_size];
    return pool->Get(0, timeout);
  }

  /**
   * @brief Release a buffer
   * @param buffer Buffer to be released
   */
  void Free(tcim::Buffer &buffer) {
    if (buffer.Size() == 0) return;
    int32_t block_size = buffer.Size();
    auto &pool = buffers_[block_size];
    pool->Free(buffer);
  }

  /**
   * @brief Get buffer pool usage statistics
   * @return Map of block size to memory usage statistics
   */
  std::map<size_t, MemoryUsage> GetStats() {
    std::map<size_t, MemoryUsage> stats;
    for (size_t block_size : block_sizes_) {
      stats[block_size] = buffers_[block_size]->GetUsage();
    }
    return stats;
  }

 private:
  /**
   * @brief Constructor
   * @param cfgs Vector of memory block configurations
   * @param mem_type Type of memory
   * @param device_id Device ID (default 0)
   */
  BufferPoolSingleton(std::vector<BufferPoolCfg> &cfgs, memType_t mem_type,
                      int32_t device_id = 0) {
    // Parameter validation
    if (cfgs.empty()) {
      throw std::runtime_error("[BufferPool] BufferPool: cfgs is empty");
    }
    if (mem_type != DRM && mem_type != HOST && mem_type != RESERVED) {
      throw std::runtime_error("[BufferPool] Invalid memType");
    }
    for (auto &cfg : cfgs) {
      if (cfg.size <= 0) {
        throw std::runtime_error(
            "[BufferPool] BufferPoolCfg size must be greater than 0");
      }
    }
    // Sort in ascending order
    std::sort(cfgs.begin(), cfgs.end(),
              [](const BufferPoolCfg &a, const BufferPoolCfg &b) {
                return ALIGN4K(a.size) < ALIGN4K(b.size);
              });
    for (int i = 0; i < cfgs.size() - 1; ++i) {
      auto &cfg0 = cfgs[i + 0];
      auto &cfg1 = cfgs[i + 1];
      if (cfg0.size % 4096 != 0 || cfg1.size % 4096 != 0) {
        throw std::runtime_error(
            "[BufferPool] BufferPoolCfg size must be aligned to 4096");
      }
      if (cfg0.size == cfg1.size) {
        throw std::runtime_error(
            "[BufferPool] BufferPoolCfg size must be not equal");
      }
    }
    buffers_.clear();
    block_sizes_.clear();
    for (auto &cfg : cfgs) {
      size_t block_size = ALIGN4K(cfg.size);
      std::unique_ptr<BufferPoolImpl> pool =
          std::make_unique<BufferPoolImpl>(cfg, mem_type);
      block_sizes_.emplace_back(block_size);
      // Create mapping between block_size and id
      buffers_[block_size] = std::move(pool);
    }
  }

 private:
  std::vector<size_t> block_sizes_;
  std::unordered_map<int32_t, std::unique_ptr<BufferPoolImpl>> buffers_;

  static inline std::unique_ptr<BufferPoolSingleton> instance_;
  static inline std::once_flag init_flag_;
};