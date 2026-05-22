/*
 * Copyright (c) 2022 HOUMO AI
 *
 * File: stream_engine.hpp
 * Description:
 *   Stream Engine Header File - Defines the StreamEngine class for managing
 * asynchronous execution of model inference tasks using multiple streams.
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
#ifndef STREAM_ENGINE_HPP
#define STREAM_ENGINE_HPP
#include <unistd.h>

#include <cassert>
#include <condition_variable>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "logging.h"
#include "tcim/tcim_runtime.h"

typedef struct RunTask {
  tcim::Module module;
  tcim::Module::RunOption option;
  std::shared_ptr<std::mutex> mutex;
  std::shared_ptr<std::condition_variable> cv;
  bool is_end = false;

  RunTask() : is_end(true) {}

  RunTask(tcim::Module& m, const tcim::Module::RunOption& opt)
      : module(m), option(opt) {
    mutex = std::make_shared<std::mutex>();
    cv = std::make_shared<std::condition_variable>();
  }
} RunTask;

typedef struct StreamQueue {
  std::queue<RunTask> queue;
  std::shared_ptr<std::mutex> mutex;
  std::shared_ptr<std::condition_variable> cv;

  StreamQueue() {
    mutex = std::make_shared<std::mutex>();
    cv = std::make_shared<std::condition_variable>();
  }
} StreamQueue;

class StreamEngine {
 public:
  StreamEngine(int stream_num) {
    stream_queues_.resize(stream_num);
    wait_counts_.resize(stream_num);
    for (int i = 0; i < stream_num; i++) {
      threads_.push_back(std::thread(&StreamEngine::StreamThread, i,
                                     std::ref(stream_queues_[i])));
    }
  }

  ~StreamEngine() {
    for (int i = 0; i < stream_queues_.size(); i++) {
      std::unique_lock<std::mutex> queue_lock(*stream_queues_[i].mutex);
      RunTask end_task;
      stream_queues_[i].queue.push(end_task);
      stream_queues_[i].cv->notify_one();
      queue_lock.unlock();
      threads_[i].join();
    }
  }

  void RunSync(tcim::Module& module, const tcim::Module::RunOption& option =
                                         tcim::Module::RunOption()) {
    int stream_id = 0;
    for (int i = 1; i < wait_counts_.size(); i++) {
      if (wait_counts_[i] < wait_counts_[stream_id]) {
        stream_id = i;
      }
    }

    RunTask task(module, option);
    std::unique_lock<std::mutex> run_lock(*task.mutex);
    std::unique_lock<std::mutex> queue_lock(*stream_queues_[stream_id].mutex);
    wait_counts_[stream_id]++;
    stream_queues_[stream_id].queue.push(task);
    stream_queues_[stream_id].cv->notify_one();
    queue_lock.unlock();
    task.cv->wait(run_lock);

    std::unique_lock<std::mutex> count_lock(*stream_queues_[stream_id].mutex);
    wait_counts_[stream_id]--;
  }

  static void StreamThread(int id, StreamQueue& qin) {
    tcim::Stream stream;
    while (1) {
      std::unique_lock<std::mutex> queue_lock(*qin.mutex);
      while (qin.queue.empty()) {
        qin.cv->wait(queue_lock);
      }
      auto task = qin.queue.front();
      if (task.is_end) {
        queue_lock.unlock();
        break;
      }
      qin.queue.pop();
      queue_lock.unlock();

      task.module.SetStream(stream);
      task.module.Run(false, task.option);
      // task.module.Run();
      task.module.Sync();

      std::lock_guard<std::mutex> run_lock(*task.mutex);
      task.cv->notify_one();
    }
    LOG_DEBUG("StreamThread {} exit.", id);
  }

 protected:
  std::vector<StreamQueue> stream_queues_;  // StreamQueue
  std::vector<int> wait_counts_;            // the num of wait tasks
  std::vector<std::thread> threads_;        // stream threads
};

#endif  // STREAM_ENGINE_HPP