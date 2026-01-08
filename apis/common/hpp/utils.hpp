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

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>

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

#endif  // __API_COMMON_HPP_UTILS_HPP__
