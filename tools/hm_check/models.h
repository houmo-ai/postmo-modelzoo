// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
#pragma once
#include <cstddef>
#include <string>

typedef struct EmbeddedModel {
    const char *name;           // "model1"
    const unsigned char *data;  // address of binary data
    size_t size;
    float num_ops{0};            // in TOPs
    int64_t read_data_size{0};   // in bytes
    int64_t write_data_size{0};  // in bytes
} embeddedModel_t;

const EmbeddedModel *get_all_models(size_t &count);
const EmbeddedModel *get_model(const std::string &name);
