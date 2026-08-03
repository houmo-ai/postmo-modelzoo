#!/usr/bin/env bash
# Copyright (c) 2026 HOUMO AI
#
# File: build_ndk.sh
# Description:
#   Build Qwen3-TTS C++ streaming demo for Android arm64-v8a.
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

if [ -e build_ndk ];then
  rm -rf build_ndk
  mkdir build_ndk
fi
set -e
if [ $(uname -s) = "Linux" ] && [ $(uname -m) = "x86_64" ]; then
  WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
  cd "${WORK_PATH}" || exit 1
  # export TCIM_RUNTIME_PATH=/hmdd/imodelzoo/tools/llm_perf/houmo-tcim-runtime-xh2
  ## if user not set NDK_PATH, use default path
  if [ x${NDK_PATH} == x ]; then
    export NDK_PATH=${WORK_PATH}/toolchains/android-ndk-r28c
    if [ -e ${NDK_PATH} ]; then
      export NDK_PATH=${WORK_PATH}/toolchains/android-ndk-r28c
    else
      echo "NDK_PATH not set, please set NDK_PATH to android-ndk-r28c"
      echo "Please Download NDK from [https://developer.android.google.cn/ndk/downloads/index.html?hl=ro]"
      echo "unzip ndk package and set NDK_PATH to android-ndk-r28c"
      exit 1
    fi
  fi

  RUN() {
    echo -e "Executing: \033[34m$*\033[0m"
    if ! "$@"; then
      echo -e "Failed: \033[31m$*\033[0m"
      exit 1
    else
      echo -e "Succeed: \033[32m$*\033[0m"
    fi
  }

  script_path=$(realpath "$0")
  project_dir=$(dirname "$script_path")
  echo "Project dir: ${project_dir}"

  if [ -n "$1" ]; then
    cmake_build_type="$1"
  else
    cmake_build_type="release"
  fi

  capitalized_build_type=$(echo "${cmake_build_type}" | perl -pe 's/\b(\w)(\w*)/\U$1\E\L$2/g')
  echo "Cmake build type: ${capitalized_build_type}"

  ## build by ndk
  build_dir=${project_dir}/build_ndk/$(echo "${cmake_build_type}" | perl -pe 'tr/A-Z/a-z/')
  mkdir -p ${build_dir}
  pushd ${build_dir}
  echo "Building in ${build_dir} ..."

  output_dir="${project_dir}/../android"
  echo "Install output to ${output_dir} ..."
  mkdir -p ${output_dir}
  RUN cmake -G Ninja ${project_dir}  \
      -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
      -DANDROID_ABI=arm64-v8a \
      -DANDROID_PLATFORM=android-35 \
      -DANDROID_NDK=${NDK_PATH} \
      -DCMAKE_INSTALL_PREFIX=${output_dir} \
      -DCMAKE_BUILD_TYPE=${capitalized_build_type} "${@:2}"

  RUN cmake --build . --target install -j 16

  popd
else
  echo "Unsupported Cross Make Platform!"
fi