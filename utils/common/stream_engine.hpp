// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file stream_engine.hpp
 */
#include <unistd.h>
#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>

#include "tcim/tcim_runtime.h"


typedef struct RunTask {
  tcim::Module module;
  tcim::Module::RunOption option;
  std::shared_ptr<std::mutex> mutex;
  std::shared_ptr<std::condition_variable> cv;
  bool is_end = false;

  RunTask() : is_end(true) {}

  RunTask(tcim::Module& m, const tcim::Module::RunOption& opt) : module(m), option(opt) {
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
      threads_.push_back(std::thread(&StreamEngine::StreamThread, i, std::ref(stream_queues_[i])));
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

  void RunSync(tcim::Module& module, const tcim::Module::RunOption& option = tcim::Module::RunOption()) {
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
    while(1) {
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
      task.module.Run();
      task.module.Sync();

      std::lock_guard<std::mutex> run_lock(*task.mutex);
      task.cv->notify_one();
    }
    // std::cout << "StreamThread " << id << " exit." << std::endl;
  }

 protected:
  std::vector<StreamQueue> stream_queues_; // stream队列
  std::vector<int> wait_counts_; // 等待运行的任务数量
  std::vector<std::thread> threads_; // stream线程
};
