// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file threads.hpp
 */
#include <unistd.h>
#include <cassert>
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

