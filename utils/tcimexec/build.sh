#!/usr/bin/env bash

if [[ -f "${HOUMO_PATH}/include/hm800_hal.h" ]]; then
  CMAKE_EXA_OPT="-DHAL=1"
fi
cmake ${CMAKE_EXA_OPT} .
make
