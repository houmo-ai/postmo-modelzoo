// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
/*!
 * \file main.cc
 */
#include <dlfcn.h>
#include <iostream>
typedef void (*run)();
int main() {
  void *handle = dlopen("./libhdpl_backbone_run.so", RTLD_LAZY);
  if (!handle) {
    std::cout << "dlopen error" << std::endl;
    return 0;
  }
  std::cout << "dlopen success" << std::endl;
  auto feat = (run)dlsym(handle, "_Z11HDPLRuntimev");
  (*feat)();
  dlclose(handle);
  return 0;
}
