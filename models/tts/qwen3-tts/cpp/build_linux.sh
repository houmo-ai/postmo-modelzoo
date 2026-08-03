#!/usr/bin/env bash
# Copyright (c) 2026 HOUMO AI
#
# File: build_linux.sh
# Description:
#   Build Qwen3-TTS C++ streaming demo for Linux x86_64 platform.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

if [ -e build ];then
  rm -rf build
  mkdir build
fi
if [ $(uname -s) = "Linux" ] &&  ([ $(uname -m) = "x86_64" ] || [ $(uname -m) = "aarch64" ]); then
  if [ "$HOUMO_TARGET" = "xh2" ]; then
    set -e

    WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
    cd "${WORK_PATH}" || exit 1

    mkdir -p build
    cd build || exit 1

    cmake -DCMAKE_INSTALL_PREFIX=$WORK_PATH/../bin -DCMAKE_BUILD_TYPE=Release ..
    make -j$(nproc)
    make install
  else
    echo "UnSupport Backend!"
  fi
else
  echo "UnSupport PlatForm!"
fi