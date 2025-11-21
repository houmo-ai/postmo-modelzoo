cmake_minimum_required(VERSION 3.16.3)
project(example_video_detect)

option(ENABLE_RESIZER "Enable resizer to process decoded images automatically" ON)
option(ENABLE_HM_ENCODER "Enable Houmo Encoder" ON)
option(ENABLE_RK_DECODER "Enable rockchip decoder" OFF)
option(ENABLE_JPEG "Enable jpeg encoder and decoder" OFF)
option(ENABLE_GEN_IMGS "Enable generation of detection result images" ON)
option(ENABLE_DECODER_ONLY "Enable decoder only" OFF)
if(ENABLE_DECODER_ONLY)
  set(ENABLE_RESIZER OFF CACHE BOOL "Option ENABLE_RESIZER" FORCE)
  set(ENABLE_GEN_IMGS OFF CACHE BOOL "Option ENABLE_GEN_IMGS" FORCE)
endif()
if(NOT ENABLE_GEN_IMGS)
  set(ENABLE_HM_ENCODER OFF CACHE BOOL "Option ENABLE_HM_ENCODER" FORCE)
endif()
if(ENABLE_HM_ENCODER)
  set(ENABLE_GEN_IMGS ON CACHE BOOL "Option ENABLE_GEN_IMGS" FORCE)
endif()
if(ENABLE_JPEG)
  set(ENABLE_RK_DECODER OFF CACHE BOOL "Option ENABLE_RK_DECODER" FORCE)
endif()
message(STATUS "ENABLE_RESIZER: ${ENABLE_RESIZER}")
message(STATUS "ENABLE_GEN_IMGS: ${ENABLE_GEN_IMGS}")
message(STATUS "ENABLE_HM_ENCODER: ${ENABLE_HM_ENCODER}")
message(STATUS "ENABLE_RK_DECODER: ${ENABLE_RK_DECODER}")
message(STATUS "ENABLE_JPEG: ${ENABLE_JPEG}")
message(STATUS "ENABLE_DECODER_ONLY: ${ENABLE_DECODER_ONLY}")

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
  if(ENABLE_HM_ENCODER)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENC_TASK)
  endif()
  if(ENABLE_GEN_IMGS)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GEN_IMGS)
  endif()
  if(ENABLE_DECODER_ONLY)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE DECODER_ONLY)
  endif()
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
  if(ENABLE_HM_ENCODER)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE ENC_TASK)
  endif()
  if(ENABLE_GEN_IMGS)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE GEN_IMGS)
  endif()
  if(ENABLE_DECODER_ONLY)
    target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE DECODER_ONLY)
  endif()
endif()

if(ENABLE_RK_DECODER)
  target_link_libraries(${CMAKE_PROJECT_NAME} ${MPP_LIBRARY})
endif()
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} ${HAL_CODEC_LIBRARY} ${HAL_LIBRARY} ${OpenCV_LIBS} avformat avcodec avutil)

if(MSVC)
else()
  target_link_libraries(${CMAKE_PROJECT_NAME} pthread dl gomp stdc++fs)
endif()

install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)
