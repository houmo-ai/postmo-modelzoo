/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: models.h
 * Description:
 *   HM System Check Models Header File - Defines the EmbeddedModel structure
 * and function declarations for accessing embedded model data used in
 * system performance testing.
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
#pragma once
#include <string>

typedef struct EmbeddedModel {
    const char *name;  // "model1"
    const void *data;  // address of binary data
    size_t size;
    float num_ops{0};            // in TOPs
    int64_t read_data_size{0};   // in bytes
    int64_t write_data_size{0};  // in bytes
} embeddedModel_t;

const EmbeddedModel *get_model(const std::string &name);
