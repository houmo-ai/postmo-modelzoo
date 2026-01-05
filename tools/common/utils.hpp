/*
 * Copyright (c) 2022 HOUMO AI
 *
 * File: utils.hpp
 * Description:
 *   Utility Functions Header File - Defines utility functions for file
 * operations and time measurement utilities.
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
#ifndef _TOOLS_COMMON_UTILS_HPP_
#define _TOOLS_COMMON_UTILS_HPP_
#include <unistd.h>

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>

#define GET_TIME() std::chrono::system_clock::now()
#define GET_COST(start, end) \
  std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()

class Utils {
 public:
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
#endif  // _TOOLS_COMMON_UTILS_HPP_