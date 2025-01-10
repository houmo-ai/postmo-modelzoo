// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file utils.hpp
 */

#include <iostream>
#include <sstream>
#include <string>

#define GET_TIME() std::chrono::system_clock::now()
#define GET_COST(start, end) std::chrono::duration_cast<std::chrono::microseconds>(end - start).count()


class Utils {
 public:
  static int ReadFile(const char *fileName, char **fileData, int *fileLen) 
  {
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
      printf("readSize(%ld) != fileSize(%ld), read %s failed!\n", readSize, fileSize, fileName);
      fclose(file);
      return -1;
    }
    *fileLen = fileSize;
    fclose(file);
    return 0;
  }

  static int WriteFile(const char *fileName, char *fileData, int fileLen) 
  {
    FILE *file = fopen(fileName, "wb"); 
    if (file == NULL) {
      perror("open file failed\n");
      return -1;
    }
    long writeSize = fwrite(fileData, 1, fileLen, file);
    if (writeSize != fileLen) {
      printf("writeSize(%ld) != fileLen(%d), write %s failed!\n", writeSize, fileLen, fileName);
      fclose(file);
      return -1;
    }
    fclose(file);
    return 0;
  }
};

