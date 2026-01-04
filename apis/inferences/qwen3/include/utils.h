/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: utils.h
 * Description:
 *   Utility functions and structures for common operations.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef __UTILS_H__
#define __UTILS_H__

#include <cassert>
#include <chrono>
#include <codecvt>
#include <cstring>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>
#include <fstream>
#include <half.hpp>
#include <iostream>
#include <locale>
#include <string>
#include <tokenizers_cpp.h>
#include <vector>

using half_float::half;

/**
 * @brief Structure to represent a message with role and content
 */
struct Message {
    std::string role;     // Role of the message sender
    std::string content;  // Content of the message
};

/**
 * @brief Load binary data from a file
 *
 * @param path Path to the file
 * @return std::string Binary data read from the file
 */
static std::string
LoadBytesFromFile(const std::string &path) {
    std::ifstream fs(path, std::ios::in | std::ios::binary);
    if (fs.fail()) {
        std::cerr << "Cannot open " << path << std::endl;
        exit(1);
    }
    std::string data;
    fs.seekg(0, std::ios::end);
    size_t size = static_cast<size_t>(fs.tellg());
    fs.seekg(0, std::ios::beg);
    data.resize(size);
    fs.read(data.data(), size);
    return data;
}

/**
 * @brief Print token IDs to standard output
 *
 * @param ids Vector of token IDs
 */
static void PrintEncodeResult(const std::vector<int> &ids) {
    std::cout << "tokens=[";
    for (size_t i = 0; i < ids.size(); ++i) {
        if (i != 0)
            std::cout << ", ";
        std::cout << ids[i];
    }
    std::cout << "]" << std::endl;
}

/**
 * @brief Read embedding weights from a binary file
 *
 * @tparam T Type of data to read
 * @param path Path to the binary file
 * @param n_elems_align Number of empty elements to pad at the end
 * @return std::unique_ptr<T[]> Unique pointer to the loaded data, or nullptr on failure
 */
template <typename T>
std::unique_ptr<T[]> readEmbeddingWeight(const std::string &path,
                                         size_t n_elems_align = 0) {
    std::ifstream ifs(path, std::ios::binary);
    if (!ifs) {
        return nullptr;  // Failed to open file
    }

    ifs.seekg(0, std::ios::end);
    const std::size_t n_bytes = ifs.tellg();
    ifs.seekg(0);

    const std::size_t n_elem = n_bytes / sizeof(T) + n_elems_align;
    auto ptr = std::make_unique<T[]>(n_elem);  // Automatically deallocated
    ifs.read(reinterpret_cast<char *>(ptr.get()), n_bytes);
    ifs.close();
    memset(reinterpret_cast<char *>(ptr.get()) + n_bytes, 0, n_elems_align * sizeof(T));
    return ptr;
}

/**
 * @brief Write embedding weights to a binary file
 *
 * @tparam T Type of data to write
 * @param path Path to the output file
 * @param embedding_weights Pointer to the data array
 * @param n Number of elements to write
 * @return bool True on success, false on failure
 */
template <typename T>
static bool writeEmbeddingWeight(const std::string &path,
                                 const T *embedding_weights,
                                 size_t n) {
    if (!embedding_weights || n == 0)
        return false;

    std::ofstream ofs(path, std::ios::binary);
    if (!ofs)
        return false;

    ofs.write(reinterpret_cast<const char *>(embedding_weights),
              n * sizeof(T));
    return ofs.good();
}

/**
 * @brief Calculate the length of a UTF-8 string in UTF-32 encoding
 *
 * @param u8 UTF-8 string view
 * @return std::size_t Length of the string in UTF-32 characters
 */
static std::size_t utf8_len(std::string_view u8) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.from_bytes(u8.data(), u8.data() + u8.size()).size();
}

/**
 * @brief Convert a UTF-8 string to UTF-32 string
 *
 * @param u8 UTF-8 string
 * @return std::u32string Converted UTF-32 string
 */
static std::u32string utf8_to_u32(const std::string &u8) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.from_bytes(u8);
}

/**
 * @brief Convert a UTF-32 string to UTF-8 string
 *
 * @param u32 UTF-32 string
 * @return std::string Converted UTF-8 string
 */
static std::string u32_to_utf8(const std::u32string &u32) {
    std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
    return conv.to_bytes(u32);
}

/**
 * @brief Check if a Unicode code point is a valid character (CJK or ASCII letters)
 *
 * @param cp Unicode code point
 * @return bool True if the code point is valid, false otherwise
 */
static bool is_valid_char(char32_t cp) noexcept {
    return
        // CJK Unified Ideographs
        (cp >= 0x4E00u && cp <= 0x9FFFu) ||
        (cp >= 0x3400u && cp <= 0x4DBFu) ||
        (cp >= 0x20000u && cp <= 0x2A6DFu) ||
        (cp >= 0x2A700u && cp <= 0x2B73Fu) ||
        (cp >= 0x2B740u && cp <= 0x2B81Fu) ||
        (cp >= 0x2B820u && cp <= 0x2CEAFu) ||
        // CJK Compatibility Ideographs
        (cp >= 0xF900u && cp <= 0xFAFFu) ||
        (cp >= 0x2F800u && cp <= 0x2FA1Fu) ||
        // ASCII Letters
        (cp >= 0x0041u && cp <= 0x005Au) ||  // A-Z
        (cp >= 0x0061u && cp <= 0x007Au);    // a-z
}

/**
 * @brief Compute the index of the maximum value in an array using Eigen library
 *
 * @tparam T Type of data in the array
 * @param ptr Pointer to the array
 * @param n Number of elements in the array
 * @return int Index of the maximum value
 */
template <typename T>
static int eigen_argmax(const T *ptr, std::size_t n) {
    using Eigen::Tensor;
    using Eigen::TensorMap;

    TensorMap<Tensor<const T, 1>> tm(static_cast<const T *>(ptr), n);

    Eigen::Tensor<Eigen::Index, 0> t = tm.argmax();
    Eigen::Index idx = t(0);

    return static_cast<int>(idx);
}

#endif  // __UTILS_H__