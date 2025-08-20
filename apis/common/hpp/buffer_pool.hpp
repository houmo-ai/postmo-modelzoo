#pragma once

#include "tcim/tcim_runtime.h"
#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <iostream>
#include <list>
#include <map>
#include <mutex>
#include <queue>
#include <thread>
#include <unistd.h>
#include <unordered_map>
#include <vector>

#define BUFFER_POOL_MAX_SIZE 300 * 1024 * 1024  // 300MB
#define ALIGN4K(size) (((size) + 4095) & ~4095)

typedef enum {
    DRM = 0,
    HOST,
    RESERVED,
} memType_t;

typedef struct BufferPoolCfg {
    size_t size;  // 块大小, 需要4K对齐，且不可重复，4k/8k/16k/32k/64k/128k/256k/512k/1M/2M/4M/8M
    int32_t num;  // 块数量
} bufferPoolCfg_t;

typedef struct MemoryUsage {
    size_t block_size;         // 块大小
    int32_t total_blocks;      // 总块数
    int32_t allocated_blocks;  // 已分配的块数
    int32_t free_blocks;       // 空闲块数
} memoryUsage_t;

// 单块固定大小内存池，只提供固定块
class BufferPoolImpl {
public:
    ~BufferPoolImpl() = default;
    // 删除拷贝构造和赋值操作
    BufferPoolImpl(const BufferPoolImpl &) = delete;
    BufferPoolImpl &operator=(const BufferPoolImpl &) = delete;
    /**
     * @brief 初始化函数
     * @param block_size 内存块大小
     * @param block_num 内存块数量
     * @param mem_type 内存类型
     */
    BufferPoolImpl(BufferPoolCfg &cfg, memType_t mem_type, int32_t device_id = 0) {
        block_size_ = ALIGN4K(cfg.size);
        block_num_ = cfg.num;
        total_size_ = static_cast<int64_t>(block_size_) * block_num_;

        if (total_size_ > BUFFER_POOL_MAX_SIZE) {
            throw std::runtime_error("[BufferPool] BufferPoolImpl: total_size_ > BUFFER_POOL_MAX_SIZE");
        }

        device_id_ = device_id;
        if ((mem_type == DRM || mem_type == RESERVED) && (device_id < 0 || device_id >= tcim::GetDeviceNum())) {
            throw std::runtime_error("[BufferPool] BufferPoolImpl: Invalid device_id");
        }
        switch (mem_type) {
        case DRM:  // DRM
            buf_ = tcim::Buffer::CreateDeviceBuffer(total_size_, device_id_, "", "");
            break;
        case HOST:  // HOST
            buf_ = tcim::Buffer::CreateHostBuffer(total_size_);
            break;
        case RESERVED:  // RESERVED
            buf_ = tcim::Buffer::CreateDeviceBuffer(total_size_, device_id_, "", "reserved");
            break;
        default:
            buf_ = tcim::Buffer();
            throw std::runtime_error("[BufferPool] Invalid memType");
        }
        // 初始化一个完整的空闲块（覆盖整个内存池）
        for (int i = 0; i < block_num_; ++i) {
            auto block_buf = buf_.GetSubBuffer(block_size_, i * block_size_);
            free_bufs_.push(block_buf);
            block_free_status_[ptr_to_string(block_buf.Data())] = true;
        }
    }
    /**
     * @brief 申请一块内存
     * @param size 申请内存的大小，默认为0，表示申请一个完整的块
     * @param timeout 申请内存的超时时间，默认认为0，表示一直等待
     */
    tcim::Buffer Get(size_t size = 0, int32_t timeout = 0) {
        int blocks_needed = size <= 0 ? 1 : (size + block_size_ - 1) / block_size_;
        if (blocks_needed > block_num_) {
            printf("[BufferPool]: request size %ld is too large\n", size);
            return tcim::Buffer();  // 请求过大
        }

        tcim::Buffer buffer;
        float wait_time = 0;  // ms
        do {
            // 从空闲队列取一个，没有就按timeout时间等待
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
                usleep(1000);  // 延时1ms
            }
        } while (wait_time < timeout || timeout == 0);
        printf("[BufferPool] Not found free block yet\n");
        return buffer;  // 空间不足
    }
    /**
     * @brief 释放buffer
     * @param buffer
     */
    void Free(const tcim::Buffer &buffer) {
        if (buffer.Size() == 0) {
            return;
        }
        std::string key = ptr_to_string(buffer.Data());
        rw_mtx_.lock();
        // 外部创建的buffer，不处理
        if (block_free_status_.find(key) == block_free_status_.end()) {
            rw_mtx_.unlock();
            printf("[BufferPool] Buffer is external\n");
            return;
        }
        // 已经释放，重复释放直接return
        if (block_free_status_[key]) {
            rw_mtx_.unlock();
            printf("[BufferPool] Buffer has been freed\n");
            return;
        }
        block_free_status_[key] = true;
        free_bufs_.push(buffer);
        allocated_blocks_--;  // 释放时减少计数
        rw_mtx_.unlock();
    }
    /**
     *
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
    int32_t allocated_blocks_ = 0;  // 新增统计变量
    int32_t block_size_ = 0;
    int32_t block_num_ = 0;
    int32_t device_id_ = 0;
    int64_t total_size_ = 0;
    tcim::Buffer buf_;   // 总内存
    std::mutex rw_mtx_;  // 内存操作锁
    std::unordered_map<std::string, bool> block_free_status_;
    std::queue<tcim::Buffer> free_bufs_;
};

// 多块大小内存池
class BufferPool {
public:
    // 删除拷贝构造和赋值操作
    BufferPool(const BufferPool &) = delete;
    BufferPool &operator=(const BufferPool &) = delete;
    /**
     * @brief 构造函数
     * @param cfgs 内存块配置
     * @param mem_type 内存类型
     */
    BufferPool(std::vector<BufferPoolCfg> &cfgs, memType_t mem_type, int32_t device_id = 0) {
        // 参数违法检查
        if (cfgs.empty()) {
            throw std::runtime_error("[BufferPool] BufferPool: cfgs is empty");
        }
        if (mem_type != DRM && mem_type != HOST && mem_type != RESERVED) {
            throw std::runtime_error("[BufferPool] Invalid memType");
        }
        for (auto &cfg : cfgs) {
            if (cfg.size <= 0) {
                throw std::runtime_error("[BufferPool] BufferPoolCfg size must be greater than 0");
            }
        }
        // 升序排序
        std::sort(cfgs.begin(), cfgs.end(), [](const BufferPoolCfg &a, const BufferPoolCfg &b) {
            return ALIGN4K(a.size) < ALIGN4K(b.size);
        });
        for (int i = 0; i < cfgs.size() - 1; ++i) {
            auto &cfg0 = cfgs[i + 0];
            auto &cfg1 = cfgs[i + 1];
            if (cfg0.size % 4096 != 0 || cfg1.size % 4096 != 0) {
                throw std::runtime_error("[BufferPool] BufferPoolCfg size must be aligned to 4096");
            }
            if (cfg0.size == cfg1.size) {
                throw std::runtime_error("[BufferPool] BufferPoolCfg size must be not equal");
            }
        }
        buffers_.clear();
        block_sizes_.clear();
        for (auto &cfg : cfgs) {
            size_t block_size = ALIGN4K(cfg.size);
            std::unique_ptr<BufferPoolImpl> pool = std::make_unique<BufferPoolImpl>(cfg, mem_type);
            block_sizes_.emplace_back(block_size);
            buffers_[block_size] = std::move(pool);  // 建立block_size和id的映射关系
        }
    }

    /**
     * @brief 申请buffer
     * @param size 申请的buffer大小
     * @param timeout 申请内存的超时时间，默认认为0，表示一直等待
     */
    tcim::Buffer Malloc(size_t size, int timeout = 0) {
        auto it = std::lower_bound(block_sizes_.begin(), block_sizes_.end(), ALIGN4K(size));
        if (it == block_sizes_.end()) {
            printf("[BufferPool] No suitable block size found, and expect size: %d\n", size);
            return tcim::Buffer();
        }
        int32_t block_size = *it;
        auto &pool = buffers_[block_size];
        return pool->Get(0, timeout);
    }

    /**
     * @brief 释放buffer
     * @param buffer
     */
    void Free(tcim::Buffer &buffer) {
        if (buffer.Size() == 0)
            return;
        int32_t block_size = buffer.Size();
        auto &pool = buffers_[block_size];
        pool->Free(buffer);
    }

    /**
     * @brief 获取buffer池使用情况
     * @return
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

// 单例多块大小内存池
class BufferPoolSingleton {
public:
    // 删除拷贝构造和赋值操作
    BufferPoolSingleton(const BufferPoolSingleton &) = delete;
    BufferPoolSingleton &operator=(const BufferPoolSingleton &) = delete;

    // 必须调用一次以初始化单例（线程安全）
    static void Init(std::vector<BufferPoolCfg> &cfgs, memType_t mem_type, int32_t device_id = 0) {
        std::call_once(init_flag_, [&]() {
            instance_.reset(new BufferPoolSingleton(cfgs, mem_type, device_id));
        });
    }

    // 获取单例引用（若未初始化将抛出）
    static BufferPoolSingleton &GetInstance() {
        if (!instance_) {
            throw std::runtime_error("[BufferPool] GetInstance called before Init");
        }
        return *instance_;
    }

    static bool IsInitialized() {
        return instance_ != nullptr;
    }

    /**
     * @brief 申请buffer
     * @param size 申请的buffer大小
     * @param timeout 申请内存的超时时间，默认认为0，表示一直等待
     */
    tcim::Buffer Malloc(size_t size, int timeout = 0) {
        auto it = std::lower_bound(block_sizes_.begin(), block_sizes_.end(), ALIGN4K(size));
        if (it == block_sizes_.end()) {
            printf("[BufferPool] No suitable block size found, and expect size: %d\n", size);
            return tcim::Buffer();
        }
        int32_t block_size = *it;
        auto &pool = buffers_[block_size];
        return pool->Get(0, timeout);
    }

    /**
     * @brief 释放buffer
     * @param buffer
     */
    void Free(tcim::Buffer &buffer) {
        if (buffer.Size() == 0)
            return;
        int32_t block_size = buffer.Size();
        auto &pool = buffers_[block_size];
        pool->Free(buffer);
    }

    /**
     * @brief 获取buffer池使用情况
     * @return
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
     * @brief 构造函数
     * @param cfgs 内存块配置
     * @param mem_type 内存类型
     */
    BufferPoolSingleton(std::vector<BufferPoolCfg> &cfgs, memType_t mem_type, int32_t device_id = 0) {
        // 参数违法检查
        if (cfgs.empty()) {
            throw std::runtime_error("[BufferPool] BufferPool: cfgs is empty");
        }
        if (mem_type != DRM && mem_type != HOST && mem_type != RESERVED) {
            throw std::runtime_error("[BufferPool] Invalid memType");
        }
        for (auto &cfg : cfgs) {
            if (cfg.size <= 0) {
                throw std::runtime_error("[BufferPool] BufferPoolCfg size must be greater than 0");
            }
        }
        // 升序排序
        std::sort(cfgs.begin(), cfgs.end(), [](const BufferPoolCfg &a, const BufferPoolCfg &b) {
            return ALIGN4K(a.size) < ALIGN4K(b.size);
        });
        for (int i = 0; i < cfgs.size() - 1; ++i) {
            auto &cfg0 = cfgs[i + 0];
            auto &cfg1 = cfgs[i + 1];
            if (cfg0.size % 4096 != 0 || cfg1.size % 4096 != 0) {
                throw std::runtime_error("[BufferPool] BufferPoolCfg size must be aligned to 4096");
            }
            if (cfg0.size == cfg1.size) {
                throw std::runtime_error("[BufferPool] BufferPoolCfg size must be not equal");
            }
        }
        buffers_.clear();
        block_sizes_.clear();
        for (auto &cfg : cfgs) {
            size_t block_size = ALIGN4K(cfg.size);
            std::unique_ptr<BufferPoolImpl> pool = std::make_unique<BufferPoolImpl>(cfg, mem_type);
            block_sizes_.emplace_back(block_size);
            buffers_[block_size] = std::move(pool);  // 建立block_size和id的映射关系
        }
    }

private:
    std::vector<size_t> block_sizes_;
    std::unordered_map<int32_t, std::unique_ptr<BufferPoolImpl>> buffers_;

    static inline std::unique_ptr<BufferPoolSingleton> instance_;
    static inline std::once_flag init_flag_;
};