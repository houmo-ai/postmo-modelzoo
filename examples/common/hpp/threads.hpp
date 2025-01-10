// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file threads.hpp
 */

#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <mutex>
#include <condition_variable>


class Barrier {
 public:
  Barrier(int dest): dest_(dest) {}

  void barrier() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_++;
    cond_.wait(lock);
  }

  void wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    while (count_ < dest_) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      lock.lock();
    }
    cond_.notify_all();
  }

  void barrier_and_wait() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_++;
    while (count_ < dest_) {
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(5));
      lock.lock();
    }
  }
  
  void reset() {
    std::unique_lock<std::mutex> lock(mtx_);
    count_ = 0;
  }

 protected:
  int count_ = 0;
  int dest_ = 0;
  std::condition_variable cond_;
  std::mutex mtx_;
};

