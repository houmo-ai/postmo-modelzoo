cmake_minimum_required(VERSION 3.16.3)
project(sample_qwen3_asr)

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
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_MPL2_ONLY")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")
endif()
add_compile_options(-Wno-deprecated-declarations)

# ---------------------------------------------------------------------
# Load houmo-llm-engine config (hm_engine INTERFACE library)
# ---------------------------------------------------------------------
include(${CMAKE_CURRENT_SOURCE_DIR}/hm_engine.cmake)

# ---------------------------------------------------------------------
# Sample target
# ---------------------------------------------------------------------
add_executable(sample_qwen3_asr
  sample_qwen3_asr.cc
  src/qwen3_asr_model.cc
)

target_include_directories(sample_qwen3_asr PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(sample_qwen3_asr PRIVATE hm_engine)
install(TARGETS sample_qwen3_asr DESTINATION .)