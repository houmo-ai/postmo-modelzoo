/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: models.cc
 * Description:
 *   HM System Check Models - Defines embedded model data and profiles for
 * system performance testing and bandwidth measurements.
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
#include "models.h"
#include <cstring>
#if defined(_MSC_VER)
#include <windows.h>
#endif

typedef struct ModelProfile {
    const char *name;
    float num_ops;
    size_t read_data_size;
    size_t write_data_size;
} modelProfile_t;

static const ModelProfile g_profiles[] = {
    {"xh2_compute", 0.309237645312, 0, 0},
    {"xh2_bandwidth_read", 0, 8388096, 0},
    {"xh2_bandwidth_write", 0, 0, 8388096},
};

static void fill_model_profile(EmbeddedModel &m) {
    for (const auto &p : g_profiles) {
        if (std::strcmp(m.name, p.name) == 0) {
            m.num_ops = p.num_ops;
            m.read_data_size = p.read_data_size;
            m.write_data_size = p.write_data_size;
            return;
        }
    }
    m.num_ops = 0;
    m.read_data_size = 0;
    m.write_data_size = 0;
}

#if defined(_MSC_VER)
const EmbeddedModel *get_model(const std::string &name) {
    HMODULE hModule = GetModuleHandle(nullptr);
    HRSRC hRes = FindResourceW(hModule, name.c_str(), RT_RCDATA);
    if (!hRes)
        return nullptr;

    DWORD size = SizeofResource(hModule, hRes);
    HGLOBAL hData = LoadResource(hModule, hRes);
    const void *ptr = LockResource(hData);

    static EmbeddedModel model;
    model.name = name.c_str();
    model.data = ptr;
    model.size = static_cast<size_t>(size);

    fill_model_profile(model);
    return &model;
}
#else
extern const unsigned char _binary_xh2_compute_hmm_start[];
extern const unsigned char _binary_xh2_compute_hmm_end[];

extern const unsigned char _binary_xh2_bandwidth_read_hmm_start[];
extern const unsigned char _binary_xh2_bandwidth_read_hmm_end[];

extern const unsigned char _binary_xh2_bandwidth_write_hmm_start[];
extern const unsigned char _binary_xh2_bandwidth_write_hmm_end[];

struct BinSym {
    const char *name;
    const unsigned char *begin;
    const unsigned char *end;
};

static const BinSym g_bins[] = {
    {"xh2_compute", _binary_xh2_compute_hmm_start, _binary_xh2_compute_hmm_end},

    {"xh2_bandwidth_read", _binary_xh2_bandwidth_read_hmm_start,
     _binary_xh2_bandwidth_read_hmm_end},

    {"xh2_bandwidth_write", _binary_xh2_bandwidth_write_hmm_start,
     _binary_xh2_bandwidth_write_hmm_end},
};

const EmbeddedModel *get_model(const std::string &name) {
    for (const auto &b : g_bins) {
        if (name == b.name) {
            static EmbeddedModel model;
            model.name = b.name;
            model.data = b.begin;
            model.size = size_t(b.end - b.begin);

            fill_model_profile(model);
            return &model;
        }
    }
    return nullptr;
}
#endif
