/*
 * Copyright (c) 2022 HOUMO AI
 *
 * File: threads.hpp
 * Description:
 *   Thread Synchronization Utilities Header File - Defines the Barrier class
 * for thread synchronization operations.
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
#ifndef THREADS_HPP
#define THREADS_HPP

#include <unistd.h>

#include <cassert>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>

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
  int count_ = 0;
  int dest_ = 0;
  std::condition_variable cond_, cond0_;
  std::mutex mtx_;
};
#endif  // THREADS_HPP
