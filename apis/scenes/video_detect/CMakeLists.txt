cmake_minimum_required(VERSION 3.16.3)
project(example_video_detect)

option(ENABLE_RESIZER "Enable resizer to process decoded images automatically" ON)
option(ENABLE_RK_DECODER "Enable rockchip decoder" OFF)
option(ENABLE_JPEG "Enable jpeg encoder and decoder" OFF)
if(ENABLE_JPEG)
    set(ENABLE_RK_DECODER OFF CACHE BOOL "Option ENABLE_RK_DECODER" FORCE)
endif()
message(STATUS "ENABLE_RESIZER: ${ENABLE_RESIZER}")
message(STATUS "ENABLE_RK_DECODER: ${ENABLE_RK_DECODER}")
message(STATUS "ENABLE_JPEG: ${ENABLE_JPEG}")

set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
set(HOUMO_SDK_PATH $ENV{HOUMO_SDK_PATH})
set(HOUMO_EXAMPLES_PATH $ENV{HOUMO_EXAMPLES_PATH})

find_library(TCIM_LIBRARY NAMES tcim_runtime_lite PATHS ${TCIM_RUNTIME_PATH}/lib NO_DEFAULT_PATH)
find_library(HAL_LIBRARY NAMES hm800_hal PATHS ${HOUMO_SDK_PATH}/hal/lib NO_DEFAULT_PATH)
find_library(HAL_CODEC_LIBRARY NAMES hm800_hal_codec PATHS ${HOUMO_SDK_PATH}/hal/lib NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED)

message("CMAKE_SOURCE_DIR=${CMAKE_SOURCE_DIR}")
message("OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message("TCIM_RUNTIME_PATH=${TCIM_RUNTIME_PATH}")
message("HOUMO_SDK_PATH=${HOUMO_SDK_PATH}")
message("HOUMO_EXAMPLES_PATH=${HOUMO_EXAMPLES_PATH}")

if(ENABLE_RK_DECODER)
  set(MPP_LIBRARY_DIR "/usr/local/lib/")
  find_library(MPP_LIBRARY NAMES rockchip_mpp rockchip_vpu PATHS ${MPP_LIBRARY_DIR} NO_DEFAULT_PATH)
endif()

include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_PATH}/include
  ${HOUMO_SDK_PATH}/hal/include/drm
  ${HOUMO_SDK_PATH}/hal/include/codec
  ${HOUMO_EXAMPLES_PATH}/apis/common/hpp
  ${OpenCV_INCLUDE_DIRS}
)

if(ENABLE_RK_DECODER)
  include_directories(/usr/local/include/rockchip)
  link_directories(/usr/local/lib)
endif()

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

if(ENABLE_JPEG)
  add_executable(
    ${CMAKE_PROJECT_NAME}
    jpeg_codec.cc
    detect_frames.cc
    jpeg_detect.cc
  )
elseif(ENABLE_RK_DECODER)
  add_executable(
    ${CMAKE_PROJECT_NAME}
    video_codec.cc
    detect_frames.cc
    video_detect.cc
    h264.cc
  )
  target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RK_DECODER)
else()
  add_executable(
    ${CMAKE_PROJECT_NAME}
    video_codec.cc
    detect_frames.cc
    video_detect.cc
  )
  if(ENABLE_RESIZER)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE RESIZER)
  endif()
endif()

target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY})
target_link_libraries(${CMAKE_PROJECT_NAME} ${HAL_CODEC_LIBRARY})
target_link_libraries(${CMAKE_PROJECT_NAME} ${HAL_LIBRARY})
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
if(ENABLE_RK_DECODER)
  target_link_libraries(${CMAKE_PROJECT_NAME} ${MPP_LIBRARY})
endif()
target_link_libraries(${CMAKE_PROJECT_NAME} avformat avcodec avutil)

if(MSVC)
else()
  target_link_libraries(${CMAKE_PROJECT_NAME} pthread dl gomp stdc++fs)
endif()

install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)
