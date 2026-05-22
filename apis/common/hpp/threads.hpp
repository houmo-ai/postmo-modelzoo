/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: threads.hpp
 * Description:
 *   Thread synchronization utilities including barrier implementations
 *   for coordinating multiple threads.T
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

#ifndef __APIS_COMMON_HPP_THREADS_HPP__
#define __APIS_COMMON_HPP_THREADS_HPP__

#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

/**
 * @brief A synchronization primitive that allows multiple threads to wait for
 * each other
 *
 * The Barrier class implements a synchronization mechanism where a specified
 * number of threads must reach a certain point in their execution before any of
 * them can proceed. This is useful for coordinating parallel operations across
 * multiple threads.
 */
class Barrier {
 public:
  /**
   * @brief Constructor for Barrier
   *
   * @param dest The number of threads that need to synchronize at the barrier
   */
  Barrier(int dest) : dest_(dest) {}

  /**
   * @brief Wait at the barrier until all threads arrive
   */
  void barrier() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_++;
    cond_.wait(lock);
  }

  /**
   * @brief Wait for all threads to reach the barrier without participating
   */
  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    while (count_ < dest_) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      lock.lock();
    }
    cond_.notify_all();
  }

  /**
   * @brief Combine barrier and wait functionality
   */
  void barrier_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_++;
    while (count_ < dest_) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      lock.lock();
    }
  }

  /**
   * @brief Reset the barrier to its initial state
   *
   * This method resets the internal counter to zero, allowing the barrier
   * to be reused for another synchronization cycle.
   */
  void reset() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_ = 0;
  }

 protected:
  int count_ = 0;  ///< Current number of threads that have reached the barrier
  int dest_ = 0;   ///< Target number of threads required to pass the barrier
  ///< Condition variable for thread synchronization
  std::condition_variable cond_;
  std::mutex mtx_;  ///< Mutex for protecting shared state
};

#endif  // __APIS_COMMON_HPP_THREADS_HPP__
