cmake_minimum_required(VERSION 3.16.3)
project(example_cxx_qwen3_vl)

include(CheckCXXCompilerFlag)
include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------
check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)

if(NOT SUPPORT_CXX17)
  message(FATAL_ERROR "C++17 is required to build ${PROJECT_NAME}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")

# -----------------------------------------------------------------------------
# Paths and sources
# -----------------------------------------------------------------------------
set(TOKENIZER_CPP_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tokenizers-cpp)
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/3rdparty)

set(SOURCE_FILES
  src/HmImageProcessor.cc
  src/HmQwenVLTokenizer.cc
  src/HmQwenVLInfer.cc
  src/SamplingManager.cc
  src/demo.cc
)

if(DEFINED ENV{HALF_INCLUDE_PATH})
  set(HALF_INCLUDE_PATH $ENV{HALF_INCLUDE_PATH})
elseif(DEFINED ENV{HOUMO_EXAMPLES_PATH})
  set(HALF_INCLUDE_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/hpp)
else()
  message(WARNING "HALF_INCLUDE_PATH not set, using default path")
  set(HALF_INCLUDE_PATH /usr/include/half)
endif()

if(ANDROID_ABI)
  set(LIB_PATH Android_xh2)
else()
  set(SYS_NAME "")
  if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(SYS_NAME Linux)
  else()
    message(WARNING "Current system is ${CMAKE_SYSTEM_NAME}, unsupported system")
  endif()

  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )

  set(HOUMO_TARGET "$ENV{HOUMO_TARGET}")
  set(LIB_PATH "${SYS_NAME}_${ARCH}_${HOUMO_TARGET}")
  message(STATUS "LIB_PATH: ${LIB_PATH}")
endif()

# -----------------------------------------------------------------------------
# OpenCV
# -----------------------------------------------------------------------------
# OpenCV modules used by current cpp code:
# - core (Mat/split/convertTo)
# - imgproc (resize/cvtColor)
# - imgcodecs (imread)
set(OPENCV_REQUIRED_MODULES core imgproc imgcodecs)
set(OPENCV_CONFIG_MODE "find_package(OpenCV)")
set(OPENCV_LIB_DIR "")

if(ANDROID_ABI)
  set(OPENCV_PATH ${CMAKE_SOURCE_DIR}/3rdparty/opencv)
  set(OpenCV_INCLUDE_DIRS ${OPENCV_PATH}/${LIB_PATH}/include)
  set(OPENCV_LIB_DIR ${OPENCV_PATH}/${LIB_PATH}/lib)

  if(NOT OpenCV_INCLUDE_DIRS STREQUAL "" AND NOT OPENCV_LIB_DIR STREQUAL "")
    set(OPENCV_LIBS
      ${OPENCV_LIB_DIR}/libopencv_core.so
      ${OPENCV_LIB_DIR}/libopencv_imgproc.so
      ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
    )
    set(OPENCV_CONFIG_MODE "Android env vars")
  endif()
endif()

if(NOT OpenCV_INCLUDE_DIRS OR NOT OPENCV_LIBS)
  find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_REQUIRED_MODULES})
  set(OPENCV_LIBS ${OpenCV_LIBS})
  set(OPENCV_CONFIG_MODE "find_package(OpenCV)")
endif()

message(STATUS "OpenCV config mode: ${OPENCV_CONFIG_MODE}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OPENCV_LIBS}")

# -----------------------------------------------------------------------------
# Target
# -----------------------------------------------------------------------------
add_executable(${PROJECT_NAME} ${SOURCE_FILES})

target_compile_definitions(${PROJECT_NAME} PRIVATE EIGEN_MPL2_ONLY)
target_include_directories(${PROJECT_NAME} PRIVATE
  ${HALF_INCLUDE_PATH}
  ${TOKENIZER_CPP_PATH}/${LIB_PATH}/include
  ${CMAKE_SOURCE_DIR}/include
  $ENV{HOUMO_PATH}/include
  ${EIGEN_PATH}
  ${OpenCV_INCLUDE_DIRS}
)

target_link_directories(${PROJECT_NAME} PRIVATE
  $ENV{HOUMO_PATH}/lib
  ${TOKENIZER_CPP_PATH}/${LIB_PATH}/lib
)

set(PROJECT_LINK_LIBS tokenizers_cpp tokenizers_c ${OPENCV_LIBS})

if(ANDROID_ABI)
  list(APPEND PROJECT_LINK_LIBS
    ${TCIM_LIBRARY}
    dl
  )
else()
  list(APPEND PROJECT_LINK_LIBS
    tcim_runtime_lite
    pthread
    ${CMAKE_DL_LIBS}
  )
endif()

target_link_libraries(${PROJECT_NAME} PRIVATE ${PROJECT_LINK_LIBS})

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS ${PROJECT_NAME} DESTINATION .)
