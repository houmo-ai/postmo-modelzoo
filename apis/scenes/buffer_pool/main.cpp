#include "MD5.h"
#include "buffer_pool.hpp"
#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace std::chrono;

// 模拟工作线程
void worker(BufferPool &pool, int32_t thread_id, size_t size) {
    size_t block_size = ALIGN4K(size);
    tcim::Buffer buf_host = tcim::Buffer::CreateHostBuffer(block_size);
    memset(buf_host.Data(), thread_id, buf_host.Size());
    MD5 md5;
    std::string ref = md5.GenerateMD5((unsigned char *)(buf_host.Data()), buf_host.Size());
    for (int i = 0; i < 1000; ++i) {
        high_resolution_clock::time_point t1 = high_resolution_clock::now();
        tcim::Buffer buf = pool.Malloc(size);
        high_resolution_clock::time_point t2 = high_resolution_clock::now();
        auto stats = pool.GetStats();
        auto usage = stats[block_size];
        if (buf.Size() > 0) {
            printf("[Thread %02d] Got buffer of size %08d bytes, and time: %04lldns, total_block: %03d, free_block: %03d\n",
                   thread_id, buf.Size(), duration_cast<nanoseconds>(t2 - t1).count(), usage.total_blocks, usage.free_blocks);
            // 写
            tcim::Status e = buf.CopyFromHost(buf_host.Data(), buf_host.Size());
            if (e != tcim::Status::OK) {
                printf("[Thread %d] CopyFromHost failed\n", thread_id);
                exit(-1);
            }
            memset(buf_host.Data(), 0xFF, buf_host.Size());
            // 读
            e = buf.CopyToHost(buf_host.Data(), buf_host.Size());
            if (e != tcim::Status::OK) {
                printf("[Thread %d] CopyToHost failed\n", thread_id);
                exit(-1);
            }
            std::string actual = md5.GenerateMD5((unsigned char *)(buf_host.Data()), buf_host.Size());
            // printf("[Thread %d] MD5: %s\n", thread_id, md5_host1.c_str());
            if (actual != ref) {
                printf("[Thread %02d] actual != ref, actual: %s, ref: %s, block_size: %d %d\n",
                       thread_id, actual.c_str(), ref.c_str(), buf_host.Size(), buf.Size());
                // exit(-1);
            }
            // 释放
            pool.Free(buf);
        } else {
            printf("[Thread %d] Failed to allocate buffer\n", thread_id);
        }
    }
}

int main() {

    // 配置：4K 8个，8K 4个，64K 2个
    std::vector<BufferPoolCfg> cfgs = {
        {4 * 1024, 100},        // 400K
        {8 * 1024, 100},        // 800K
        {16 * 1024, 100},       // 1600K
        {32 * 1024, 100},       // 3200K
        {64 * 1024, 32},        // 2M
        {128 * 1024, 32},       // 4M
        {256 * 1024, 32},       // 8M
        {512 * 1024, 32},       // 16M
        {1024 * 1024, 32},      // 32M
        {2 * 1024 * 1024, 32},  // 64M
        {4 * 1024 * 1024, 16},  // 64M
        {8 * 1024 * 1024, 8},   // 64M
    };

    BufferPool pool(cfgs, HOST);

    // 启动多个线程并发消费
    std::vector<std::thread>
        threads;
    for (int t = 0; t < cfgs.size(); ++t) {
        size_t block_size = cfgs[t].size;
        threads.emplace_back(worker, std::ref(pool), t * 2 + 0, block_size);
        threads.emplace_back(worker, std::ref(pool), t * 2 + 1, block_size - 1024);
    }

    for (auto &th : threads) {
        th.join();
    }

    std::cout << "All done." << std::endl;
    return 0;
}
