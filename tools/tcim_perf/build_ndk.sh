#!/usr/bin/env bash

set -e

## if user not set NDK_PATH, use default path
if [ x${NDK_PATH} == x ]; then
  export NDK_PATH=${HOUMO_EXAMPLES_PATH}/toolchains/android-ndk-r28c
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

output_dir="${project_dir}/output"
echo "Install output to ${output_dir} ..."
mkdir -p ${output_dir}

RUN cmake -G Ninja ${project_dir}  \
    -DSKIP_GEN_HEADER=ON \
    -DCMAKE_TOOLCHAIN_FILE=${NDK_PATH}/build/cmake/android.toolchain.cmake \
    -DANDROID_ABI=arm64-v8a \
    -DANDROID_PLATFORM=android-35 \
    -DANDROID_NDK=${NDK_PATH} \
    -DENABLE_XH1_HDI=OFF -DENABLE_XH2A_HAL=OFF -DENABLE_XH2_IPU=OFF -DBUILD_TESTS=OFF \
    -DCMAKE_INSTALL_PREFIX=${output_dir} -DCMAKE_BUILD_TYPE=${capitalized_build_type} ${@:2}
                    

RUN cmake --build . --target install -j 16

popd
