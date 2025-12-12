// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.

#include "models.h"

// 声明多个模型的符号
extern const unsigned char _binary_xh2_compute_hmm_start[];
extern const unsigned char _binary_xh2_compute_hmm_end[];

extern const unsigned char _binary_xh2_bandwidth_read_hmm_start[];
extern const unsigned char _binary_xh2_bandwidth_read_hmm_end[];

extern const unsigned char _binary_xh2_bandwidth_write_hmm_start[];
extern const unsigned char _binary_xh2_bandwidth_write_hmm_end[];

// 如有更多模型，继续写：
// extern const unsigned char _binary_xxx_hmm_start[];
// extern const unsigned char _binary_xxx_hmm_end[];

static EmbeddedModel g_models[] = {
    {"xh2_compute", _binary_xh2_compute_hmm_start,
     size_t(_binary_xh2_compute_hmm_end - _binary_xh2_compute_hmm_start),
     0.309237645312, 0, 0},
    {"xh2_bandwidth_read", _binary_xh2_bandwidth_read_hmm_start,
     size_t(_binary_xh2_bandwidth_read_hmm_end -
            _binary_xh2_bandwidth_read_hmm_start),
     0, 8388096, 0},
    {"xh2_bandwidth_write", _binary_xh2_bandwidth_write_hmm_start,
     size_t(_binary_xh2_bandwidth_write_hmm_end -
            _binary_xh2_bandwidth_write_hmm_start),
     0, 0, 8388096},
};

const EmbeddedModel *get_all_models(size_t &count) {
    count = sizeof(g_models) / sizeof(g_models[0]);
    return g_models;
}

const EmbeddedModel *get_model(const std::string &name) {
    for (auto &m : g_models) {
        if (m.name == name)
            return &m;
    }
    return nullptr;
}
