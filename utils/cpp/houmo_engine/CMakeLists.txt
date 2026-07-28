cmake_minimum_required(VERSION 3.16.3)
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()
project(houmo_infer)

include(CheckCXXCompilerFlag)
include("$ENV{HOUMO_EXAMPLES_PATH}/cmake/tcim_runtime.cmake")

# -----------------------------------------------------------------------------
# Platform configuration
# -----------------------------------------------------------------------------
set(HOUMO_PLATFORM_ANDROID FALSE)
set(HOUMO_PLATFORM_WINDOWS FALSE)
set(HOUMO_PLATFORM_LINUX FALSE)
set(HOUMO_PLATFORM_NAME "unknown")
set(HOUMO_NEED_PTHREAD FALSE)
set(LIB_PATH "")
set(HOUMO_TARGET_LINK_OPTIONS "")

if(ANDROID_ABI)
  set(HOUMO_PLATFORM_ANDROID TRUE)
  include($ENV{HOUMO_EXAMPLES_PATH}/cmake/platforms/android.cmake)
elseif(MSVC)
  set(HOUMO_PLATFORM_WINDOWS TRUE)
  include($ENV{HOUMO_EXAMPLES_PATH}/cmake/platforms/windows.cmake)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(HOUMO_PLATFORM_LINUX TRUE)
  include($ENV{HOUMO_EXAMPLES_PATH}/cmake/platforms/linux.cmake)
elseif(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  set(HOUMO_PLATFORM_OHOS TRUE)
  include($ENV{HOUMO_EXAMPLES_PATH}/cmake/platforms/ohos.cmake)
else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

message(STATUS "HOUMO platform: ${HOUMO_PLATFORM_NAME}")
if(LIB_PATH)
  message(STATUS "LIB_PATH: ${LIB_PATH}")
endif()

# -----------------------------------------------------------------------------
# Build options
# -----------------------------------------------------------------------------
check_cxx_compiler_flag("${HOUMO_CXX17_FLAG}" SUPPORT_CXX17)

if(NOT SUPPORT_CXX17)
  message(FATAL_ERROR "C++17 is required to build ${PROJECT_NAME}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(${HOUMO_COMPILE_OPTIONS})
if(HOUMO_RELEASE_FLAGS)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${HOUMO_RELEASE_FLAGS}")
endif()
if(HOUMO_DEBUG_FLAGS)
  set(CMAKE_CXX_FLAGS_DEBUG "${HOUMO_DEBUG_FLAGS}")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------
# Image: stb_image + pure C++ resize (header-only under 3rdparty).
# Audio: miniaudio (header-only under  3rdparty) + pure C++ mel.

# tokenizer.cpp / header-only deps under HOUMO_EXAMPLES_PATH
if(DEFINED ENV{HOUMO_EXAMPLES_PATH})
  message(STATUS "Raw env: [$ENV{HOUMO_EXAMPLES_PATH}]")
  set(THIRDPARTY_PATH $ENV{HOUMO_EXAMPLES_PATH}/3rdparty)
else()
  message(FATAL_ERROR "Environment variable HOUMO_EXAMPLES_PATH is not set!...")
endif()
if(NOT EXISTS "${THIRDPARTY_PATH}/stb/stb_image.h")
  message(FATAL_ERROR "stb_image.h not found at ${THIRDPARTY_PATH}/stb/stb_image.h")
endif()
if(NOT EXISTS "${THIRDPARTY_PATH}/miniaudio/miniaudio.h")
  message(FATAL_ERROR "miniaudio.h not found at ${THIRDPARTY_PATH}/miniaudio/miniaudio.h")
endif()
if(NOT EXISTS "${THIRDPARTY_PATH}/half/half.hpp")
  message(FATAL_ERROR "half.hpp not found at ${THIRDPARTY_PATH}/half/half.hpp")
endif()
include("$ENV{HOUMO_EXAMPLES_PATH}/cmake/tokenizer.cpp.cmake")

# -----------------------------------------------------------------------------
# Paths and sources
# -----------------------------------------------------------------------------
set(CORE_SOURCES
  src/core/version.cc
  src/core/context.cc
  src/core/vlm_model.cc
  src/core/model_factory.cc
  src/core/llm_model.cc
  src/core/asr_model.cc
)

set(MODULE_SOURCES
  src/modules/audio_processor.cc
  src/modules/tokenizer.cc
  src/modules/embedding.cc
  src/modules/sampler.cc
  src/modules/streaming_decoder.cc
  src/modules/image_processor.cc
  src/modules/perf_profiler.cc
)

# -----------------------------------------------------------------------------
# Library target
# -----------------------------------------------------------------------------
add_library(houmo_infer SHARED
  ${CORE_SOURCES}
  ${MODULE_SOURCES}
)
target_compile_definitions(houmo_infer PUBLIC EIGEN_MPL2_ONLY)

if(HOUMO_EXPORT_ALL_SYMBOLS)
  set_target_properties(houmo_infer PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(HOUMO_TARGET_LINK_OPTIONS)
  target_link_options(houmo_infer PRIVATE ${HOUMO_TARGET_LINK_OPTIONS})
endif()

target_include_directories(houmo_infer PUBLIC
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_INCLUDE}
  ${THIRDPARTY_PATH}
)

target_link_directories(houmo_infer PUBLIC
  ${TCIM_RUNTIME_PATH}/lib
)

set(HOUMO_INFER_LIBS
  ${TCIM_LIBRARY}
  tokenizer_lib
  ${CMAKE_DL_LIBS}
)

if(HOUMO_NEED_PTHREAD)
  list(APPEND HOUMO_INFER_LIBS pthread)
endif()
if(HOUMO_EXTRA_LIBS)
  list(APPEND HOUMO_INFER_LIBS ${HOUMO_EXTRA_LIBS})
endif()

target_link_libraries(houmo_infer PUBLIC ${HOUMO_INFER_LIBS})

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS houmo_infer tokenizer_lib
  RUNTIME DESTINATION ${LIB_INSTALL_PATH}
  LIBRARY DESTINATION ${LIB_INSTALL_PATH}
  ARCHIVE DESTINATION ${LIB_INSTALL_PATH}
)

# install(TARGETS tokenizer_lib
#   RUNTIME DESTINATION ${LIB_INSTALL_PATH}
#   LIBRARY DESTINATION ${LIB_INSTALL_PATH}
#   ARCHIVE DESTINATION ${LIB_INSTALL_PATH}
# )

install(FILES ${CMAKE_SOURCE_DIR}/convert_embed.py
        DESTINATION .
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
