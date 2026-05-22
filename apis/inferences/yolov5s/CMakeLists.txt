cmake_minimum_required(VERSION 3.16.3)
project(example_yolov5s)

option(ENABLE_ORT "Whether it supports the onnxruntime c++ environment" OFF)

set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
set(HOUMO_EXAMPLES_PATH $ENV{HOUMO_EXAMPLES_PATH})

find_library(TCIM_LIBRARY NAMES tcim_runtime_lite PATHS ${TCIM_RUNTIME_PATH}/lib NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED)

message(STATUS "ENABLE_ORT: ${ENABLE_ORT}")

if(ENABLE_ORT)
    set(ORT_PATH ${CMAKE_SOURCE_DIR}/../../models/3rdparty/onnxruntime)
    link_directories(${ORT_PATH}/lib)
    include_directories(${ORT_PATH}/include)
endif()

include_directories(${TCIM_RUNTIME_PATH}/include)
include_directories(${HOUMO_EXAMPLES_PATH}/apis/common/hpp)
include_directories(${OpenCV_INCLUDE_DIRS})
message("OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message("TCIM_RUNTIME_PATH=${TCIM_RUNTIME_PATH}")
message("HOUMO_EXAMPLES_PATH=${HOUMO_EXAMPLES_PATH}")

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
    add_compile_options(/utf-8)
else()
    add_definitions(-w)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(${CMAKE_PROJECT_NAME} yolov5s.cc)

if(ENABLE_ORT)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENABLE_ORT)
    target_link_libraries(${CMAKE_PROJECT_NAME} onnxruntime)
endif()
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY})

if(MSVC)
else()
    target_link_libraries(${CMAKE_PROJECT_NAME} pthread dl gomp stdc++fs)
endif()

install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)
