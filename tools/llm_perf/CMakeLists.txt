cmake_minimum_required(VERSION 3.16.3)
project(llm_perf)

include(CheckCXXCompilerFlag)

set(HOUMO_ENGINE_DIR $ENV{HOUMO_EXAMPLES_PATH}/utils/cpp/houmo_engine)
set(HOUMO_PLATFORM_ANDROID FALSE)
set(HOUMO_PLATFORM_WINDOWS FALSE)
set(HOUMO_PLATFORM_LINUX FALSE)
set(HOUMO_PLATFORM_NAME "unknown")
set(HOUMO_NEED_PTHREAD FALSE)
set(THIRDPARTY_PATH $ENV{HOUMO_EXAMPLES_PATH}/3rdparty)

if(ANDROID_ABI)
  set(HOUMO_PLATFORM_ANDROID TRUE)
  include($ENV{HOUMO_EXAMPLES_PATH}/cmake/platforms/android.cmake)
elseif(MSVC)
  set(HOUMO_PLATFORM_WINDOWS TRUE)
  include($ENV{HOUMO_EXAMPLES_PATH}/cmake/platforms/windows.cmake)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(HOUMO_PLATFORM_LINUX TRUE)
  include($ENV{HOUMO_EXAMPLES_PATH}/cmake/platforms/linux.cmake)
else()
  message(FATAL_ERROR "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

message(STATUS "HOUMO platform: ${HOUMO_PLATFORM_NAME}")

check_cxx_compiler_flag("${HOUMO_CXX17_FLAG}" SUPPORT_CXX17)
if(NOT SUPPORT_CXX17)
  message(FATAL_ERROR "C++17 is required to build ${PROJECT_NAME}")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_compile_options(${HOUMO_COMPILE_OPTIONS})
if(MSVC)
  add_compile_options(/DNOMINMAX /DEIGEN_MPL2_ONLY)
else()
  add_compile_options(-DEIGEN_MPL2_ONLY)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
if(HOUMO_RELEASE_FLAGS)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${HOUMO_RELEASE_FLAGS}")
endif()
if(HOUMO_DEBUG_FLAGS)
  set(CMAKE_CXX_FLAGS_DEBUG "${HOUMO_DEBUG_FLAGS}")
endif()

if(MSVC)
  set(ORIGINAL_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4244")
endif()
include($ENV{HOUMO_EXAMPLES_PATH}/cmake/yaml-cpp.cmake)
if(MSVC)
  set(CMAKE_CXX_FLAGS ${ORIGINAL_CXX_FLAGS})
endif()

include("$ENV{HOUMO_EXAMPLES_PATH}/cmake/tcim_runtime.cmake")
include("$ENV{HOUMO_EXAMPLES_PATH}/cmake/houmo_engine.cmake")

option(ENABLE_ASR "Enable ASR support" ON)
message(STATUS "ENABLE_ASR: ${ENABLE_ASR}")

include_directories(
  $ENV{HOUMO_SDK_PATH}/hal/export/include
  $ENV{HOUMO_SDK_PATH}/hal/include
  ${CMAKE_SOURCE_DIR}/include
  ${THIRDPARTY_PATH}
)

link_directories(
  $ENV{HOUMO_SDK_PATH}/hal/lib
)

set(SOURCE_FILES
  src/utils/perf_dumper/perf_dumper.cc
  src/utils/host_monitor/host_monitor.cc
  src/utils/device_monitor/device_monitor.cc
  src/utils/perf_tracker/inference_perf_tracker.cc
  src/utils/HmEmbedding.cc
  src/vlm/HmvllmInfer.cc
  src/llm/HmllmInfer.cc
  src/llm/HmllmInferMultiBatch.cc
  llm_perf.cc
)
if(ENABLE_ASR)
  list(APPEND SOURCE_FILES src/asr/PerfAsr.cc)
endif()

set(INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
)

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

if(ENABLE_ASR)
  target_compile_definitions(${PROJECT_NAME} PRIVATE ENABLE_ASR)
endif()

set(LLM_PERF_LIBS
  ${TCIM_LIBRARY}
  yaml-cpp
  houmo_engine
)

if(HOUMO_NEED_PTHREAD)
  list(APPEND LLM_PERF_LIBS pthread)
endif()

if(NOT MSVC)
  list(APPEND LLM_PERF_LIBS hal_xh2a ${CMAKE_DL_LIBS})
endif()

if(MSVC)
  list(APPEND LLM_PERF_LIBS libhal_xh2a ws2_32 Userenv ntdll Bcrypt kernel32)
endif()


target_link_libraries(${PROJECT_NAME} PRIVATE ${LLM_PERF_LIBS})

install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/convert_embed.py
        DESTINATION .
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
