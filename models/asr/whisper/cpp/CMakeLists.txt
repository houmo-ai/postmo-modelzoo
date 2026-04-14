cmake_minimum_required(VERSION 3.16.3)
project(whisper-demo)

include(CheckCXXCompilerFlag)
include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------
if(MSVC)
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
else()
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
endif()

if(NOT SUPPORT_CXX17)
  message(FATAL_ERROR "C++17 is required to build ${PROJECT_NAME}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/utf-8)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")
endif()

# -----------------------------------------------------------------------------
# Paths and sources
# -----------------------------------------------------------------------------
set(TOKENIZER_CPP_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tokenizers-cpp)
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/3rdparty)

set(SOURCE_FILES
  src/hm_whisper_audio.cc
  src/hm_whisper_infer.cc
  src/miniaudio_impl.cc
  src/SamplingManager.cc
  src/whisper_demo.cc
)

if(DEFINED ENV{HALF_INCLUDE_PATH})
  set(HALF_INCLUDE_PATH $ENV{HALF_INCLUDE_PATH})
elseif(DEFINED ENV{HOUMO_EXAMPLES_PATH})
  set(HALF_INCLUDE_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include)
else()
  message(WARNING "HALF_INCLUDE_PATH not set, using default path")
  set(HALF_INCLUDE_PATH /usr/include/half)
endif()

if(MSVC)
  set(LIB_PATH win11_x64)
elseif(ANDROID_ABI)
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
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/audio/include
)

target_link_directories(${PROJECT_NAME} PRIVATE
  $ENV{HOUMO_PATH}/lib
  ${TOKENIZER_CPP_PATH}/${LIB_PATH}/lib
)

set(PROJECT_LINK_LIBS tokenizers_cpp tokenizers_c)

if(MSVC)
  list(APPEND PROJECT_LINK_LIBS
    ${TCIM_LIBRARY}
    ${CMAKE_DL_LIBS}
    ws2_32
    Userenv
    ntdll
    Bcrypt
    kernel32
  )
elseif(ANDROID_ABI)
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
