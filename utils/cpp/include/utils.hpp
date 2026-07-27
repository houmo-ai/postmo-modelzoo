/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: utils.hpp
 * Description:
 *   Utility functions for file operations, timing, and tensor handling
 *   used across the APIs common components.
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

#ifndef __API_COMMON_HPP_UTILS_HPP__
#define __API_COMMON_HPP_UTILS_HPP__

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "datasets/imagenet.hpp"
#include "logging.h"
#include "tcim/tcim_runtime.h"

/**
 * @brief Macro to get current time for timing operations
 *
 * Returns a time point representing the current moment using system clock
 */
#define GET_TIME() std::chrono::system_clock::now()
/**
 * @brief Macro to calculate time difference in microseconds
 *
 * @param start Start time point
 * @param end End time point
 * @return Duration in microseconds
 */
#define GET_COST(start, end) \
  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()
/**
 * @brief Macro to convert a number to the nearest even number
 *
 * Rounds down an odd number to make it even, leaves even numbers unchanged
 *
 * @param n Input number
 * @return The nearest even number less than or equal to n
 */
#define TO_EVEN(n) (n & ~1)

/**
 * @brief Utility class providing common file operations.
 */
class Utils {
 public:
  /**
   * @brief Read contents of a file into memory
   *
   * Opens a file in binary mode and reads its contents into dynamically
   * allocated memory.
   *
   * @param fileName Path to the file to read
   * @param fileData Pointer to a character pointer that will receive the file
   * data
   * @param fileLen Pointer to an integer that will receive the file size
   * @return 0 on success, -1 on failure
   *
   * @note The caller is responsible for freeing the memory allocated for
   * fileData
   */
  static int ReadFile(const char *fileName, char **fileData, int *fileLen) {
    FILE *file = fopen(fileName, "rb");
    if (file == NULL) {
      perror("open file failed\n");
      return -1;
    }

    fseek(file, 0, SEEK_END);
    long fileSize = ftell(file);
    fseek(file, 0, SEEK_SET);

    *fileData = (char *)malloc(fileSize);
    if (*fileData == NULL) {
      printf("malloc fileData size:%ld fialed\n", fileSize);
      fclose(file);
      return -1;
    }
    long readSize = fread(*fileData, 1, fileSize, file);
    if (readSize != fileSize) {
      printf("readSize(%ld) != fileSize(%ld), read %s failed!\n", readSize,
             fileSize, fileName);
      fclose(file);
      return -1;
    }
    *fileLen = fileSize;
    fclose(file);
    return 0;
  }

  /**
   * @brief Write data to a file
   *
   * Writes the specified data to a file in binary mode.
   *
   * @param fileName Path to the file to write
   * @param fileData Pointer to the data to write
   * @param fileLen Size of the data to write
   * @return 0 on success, -1 on failure
   */
  static int WriteFile(const char *fileName, char *fileData, int fileLen) {
    FILE *file = fopen(fileName, "wb");
    if (file == NULL) {
      perror("open file failed\n");
      return -1;
    }
    long writeSize = fwrite(fileData, 1, fileLen, file);
    if (writeSize != fileLen) {
      printf("writeSize(%ld) != fileLen(%d), write %s failed!\n", writeSize,
             fileLen, fileName);
      fclose(file);
      return -1;
    }
    fclose(file);
    return 0;
  }
};

/**
 * @brief Write binary data to a file
 *
 * Writes the specified data buffer to a file in binary format.
 *
 * @param data Pointer to the data to write
 * @param size Size of the data to write
 * @param filename Path to the output file
 */
static void WriteBin2File(void *data, size_t size, const char *filename) {
  std::ofstream fs(filename, std::ios::binary);
  std::string str = std::string(static_cast<char *>(data), size);
  fs << str;
  fs.close();
}

/**
 * @brief Convert TensorInfo to string representation
 *
 * Converts a tcim::TensorInfo object to its string representation for
 * debugging and logging purposes.
 *
 * @param tensor_info The TensorInfo object to convert
 * @return String representation of the TensorInfo
 */
static inline std::string TensorInfo2Str(const tcim::TensorInfo &tensor_info) {
  std::stringstream ss;
  ss << tensor_info;
  return ss.str();
}

/**
 * @brief Convert a TensorInfo object to a human-readable string format
 *
 * This function takes a tcim::TensorInfo object and constructs a string that
 * describes its properties, such as data type, shape, and memory layout.
 *
 * @param tensor_info The TensorInfo object to convert
 * @return A string representation of the TensorInfo for logging and debugging
 */
static inline size_t GetElementCount(const tcim::TensorInfo &info) {
  size_t count = 1;
  for (auto dim : info.Shape()) {
    count *= static_cast<size_t>(dim);
  }
  return count;
}

/**
 * @brief Apply a numerically stable softmax on a float logits buffer
 *
 * This function takes a pointer to a buffer of logits and the number of
 * elements, and returns a vector of probabilities after applying the softmax
 * function.
 *
 * @param logits Pointer to the logits buffer
 * @param count Number of elements in the logits buffer
 * @return A vector of probabilities after applying softmax
 */
static inline std::vector<float> Softmax(const float *logits, size_t count) {
  std::vector<float> probs(count, 0.0f);
  if (count == 0) {
    return probs;
  }

  float max_value = std::numeric_limits<float>::lowest();
  for (size_t i = 0; i < count; ++i) {
    max_value = std::max(max_value, logits[i]);
  }

  float sum = 0.0f;
  for (size_t i = 0; i < count; ++i) {
    probs[i] = std::exp(logits[i] - max_value);
    sum += probs[i];
  }

  if (sum == 0.0f) {
    return probs;
  }

  for (auto &value : probs) {
    value /= sum;
  }
  return probs;
}

/**
 * Get the top K maximum values and their index information
 * This function sorts the value-index pairs in descending order and prints the
 * top K results
 *
 * @param topk Number of top K elements to retrieve
 * @param sort_pairs Vector of pairs containing values and indices, where T is
 * the value type and int is the original index
 * @return Returns the original index corresponding to the maximum value
 */
template <typename T>
static inline int GetTopK(int topk, std::vector<std::pair<T, int>> sort_pairs) {
  // Sort pairs in descending order by value
  std::sort(sort_pairs.begin(), sort_pairs.end(),
            [](const std::pair<T, int> &a, const std::pair<T, int> &b) {
              return a.first > b.first;
            });

  const int valid_topk = std::min(topk, static_cast<int>(sort_pairs.size()));
  if (valid_topk <= 0) {
    return -1;
  }

  // Print detailed information for top K elements, including index, confidence
  // and label
  for (int i = 0; i < valid_topk; ++i) {
    LOG_INFO("top{}: Index={} Conf={}, Label=[{}]", i + 1, sort_pairs[i].second,
             sort_pairs[i].first, Imagenet::GetLabel(sort_pairs[i].second));
  }

  return sort_pairs[0].second;
}

#endif  // __API_COMMON_HPP_UTILS_HPP__
