cmake_minimum_required(VERSION 3.16.3)
project(houmo_demo)

unset(TVM_LIBRARY CACHE)
set(HDPL_TOOLCHAIN_ITVM_INSTALL "../../../../itvm/build/install")
if(DEFINED ENV{HDPL_TOOLCHAIN_ITVM_INSTALL})
  message(STATUS "HDPL_TOOLCHAIN_ITVM_INSTALL: " $ENV{HDPL_TOOLCHAIN_ITVM_INSTALL})
  set(HDPL_TOOLCHAIN_ITVM_INSTALL $ENV{HDPL_TOOLCHAIN_ITVM_INSTALL})
endif()
find_library(TVM_LIBRARY NAMES tvm PATHS ${HDPL_TOOLCHAIN_ITVM_INSTALL} NO_DEFAULT_PATH)
if(NOT DEFINED ${TVM_LIBRARY})
  find_library(TVM_LIBRARY NAMES tvm PATHS ${HDPL_TOOLCHAIN_ITVM_INSTALL}/lib NO_DEFAULT_PATH)
endif()

if(DEFINED ENV{HOUMO_PATH})
  set(HDPL_INSTALL_PATH $ENV{HOUMO_PATH})
else()
  set(HDPL_INSTALL_PATH $ENV{HDPL_PATH})
endif()

if(NOT DEFINED ${HDPL_LIBRARY})
  find_library(HDPL_LIBRARY NAMES hdplrt PATHS ${HDPL_INSTALL_PATH}/lib NO_DEFAULT_PATH)
endif()

include_directories(${HDPL_INSTALL_PATH}/include)
include_directories(${HDPL_TOOLCHAIN_ITVM_INSTALL}/include)
message(STATUS "HDPL_TOOLCHAIN DIR: ${HDPL_TOOLCHAIN_ITVM_INSTALL}")


add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -O2")
add_executable(hdpl_resnet50_run hdpl_resnet50_run.cc)
target_link_libraries(hdpl_resnet50_run ${TVM_LIBRARY} ${HDPL_LIBRARY} -pthread dl gomp)
