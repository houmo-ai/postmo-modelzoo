cmake_minimum_required(VERSION 3.16.3)
project(houmo_infer)

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

add_compile_options(-Wno-deprecated-declarations)

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

# -----------------------------------------------------------------------------
# Dependencies
# -----------------------------------------------------------------------------

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
    set(OpenCV_FOUND TRUE)
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

# GTest for unit testing (local 3rdparty)
option(BUILD_TESTS "Build unit tests" ON)
if(BUILD_TESTS)
  set(GTEST_ROOT ${CMAKE_SOURCE_DIR}/3rdparty/googletest)
  add_subdirectory(${GTEST_ROOT} googletest)
  enable_testing()
endif()

set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/3rdparty)

# Audio 3rdparty (miniaudio, libsamplerate, kaldi-native-fbank)
set(AUDIO_3RDPARTY_PATH ${CMAKE_SOURCE_DIR}/3rdparty/audio)
set(AUDIO_3RDPARTY_INCLUDE_DIR ${AUDIO_3RDPARTY_PATH}/3rdparty_build/include)
set(AUDIO_3RDPARTY_LIBRARY_DIR ${AUDIO_3RDPARTY_PATH}/3rdparty_build/lib)

# tokenizers_cpp (HuggingFace tokenizers C++, local 3rdparty)
set(TOKENIZERS_CPP_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tokenizers-cpp)

# Determine platform-specific lib path
if(NOT ANDROID_ABI)
  set(SYS_NAME "Linux")
  execute_process(
    COMMAND uname -m
    OUTPUT_VARIABLE ARCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  set(HOUMO_TARGET "$ENV{HOUMO_TARGET}")
  set(LIB_PATH "${SYS_NAME}_${ARCH}_${HOUMO_TARGET}")
else()
  set(LIB_PATH "Android_xh2")
endif()

# -----------------------------------------------------------------------------
# Paths and sources
# -----------------------------------------------------------------------------

# Half precision floating point
if(DEFINED ENV{HALF_INCLUDE_PATH})
  set(HALF_INCLUDE_PATH $ENV{HALF_INCLUDE_PATH})
elseif(DEFINED ENV{HOUMO_EXAMPLES_PATH})
  set(HALF_INCLUDE_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include)
else()
  set(HALF_INCLUDE_PATH /usr/include/half)
endif()

# Core sources
set(CORE_SOURCES
  src/core/version.cc
  src/core/context.cc
  src/core/vlm_model.cc
  src/core/model_factory.cc
)

# Module sources
set(MODULE_SOURCES
  src/modules/audio_processor.cc
  src/modules/tokenizer.cc
  src/modules/embedding.cc
  src/modules/sampler.cc
  src/modules/streaming_decoder.cc
  src/modules/image_processor.cc
  src/modules/perf_profiler.cc
)

# Backend sources
set(BACKEND_SOURCES
  # src/backends/tcim_backend.cc  # TODO: implement later
)

# Model sources
set(MODEL_SOURCES
  src/models/llm_model.cc
  src/models/qwen3_llm_model.cc
  src/models/qwen35_mllm_model.cc
  src/models/qwen3_vlm_model.cc
  src/models/asr_model.cc
  src/models/whisper_model.cc
  src/models/glm_asr_model.cc
  src/models/qwen3_asr_model.cc
)

# Sample sources
set(SAMPLE_SOURCES
  sample/sample_infer.cc
)

# -----------------------------------------------------------------------------
# Library target
# -----------------------------------------------------------------------------
add_library(houmo_infer STATIC
  ${CORE_SOURCES}
  ${MODULE_SOURCES}
  ${BACKEND_SOURCES}
  ${MODEL_SOURCES}
)

target_include_directories(houmo_infer PUBLIC
  ${HALF_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_INCLUDE}
  ${TOKENIZERS_CPP_PATH}/${LIB_PATH}/include
  ${EIGEN_PATH}
  ${OpenCV_INCLUDE_DIRS}
  ${AUDIO_3RDPARTY_INCLUDE_DIR}
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/audio/include
)

target_link_directories(houmo_infer PUBLIC
  ${TCIM_RUNTIME_PATH}/lib
  ${TOKENIZERS_CPP_PATH}/${LIB_PATH}/lib
  ${AUDIO_3RDPARTY_LIBRARY_DIR}
)

target_link_libraries(houmo_infer PUBLIC
  tcim_runtime_lite
  tokenizers_cpp
  tokenizers_c
  kaldi-native-fbank-core
  samplerate
  ${CMAKE_DL_LIBS}
)

if(NOT ANDROID_ABI)
  target_link_libraries(houmo_infer PUBLIC pthread)
endif()

# Link OpenCV if available
if(OpenCV_FOUND)
  target_link_libraries(houmo_infer PUBLIC ${OPENCV_LIBS})
  target_compile_definitions(houmo_infer PUBLIC HOUOMO_USE_OPENCV=1)
endif()

# -----------------------------------------------------------------------------
# Sample targets
# -----------------------------------------------------------------------------


add_executable(sample_infer sample/sample_infer.cc)
# whole-archive makes sure all symbols in houmo_infer are exported for dynamic loading in sample_infer
target_link_libraries(sample_infer PRIVATE
  -Wl,--whole-archive houmo_infer -Wl,--no-whole-archive
)
target_include_directories(sample_infer PRIVATE
  ${HALF_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_INCLUDE}
  ${TOKENIZERS_CPP_PATH}/${LIB_PATH}/include
  ${EIGEN_PATH}
  ${OpenCV_INCLUDE_DIRS}
)

# Whisper ASR sample
add_executable(sample_whisper_asr sample/sample_whisper_asr.cc)
target_link_libraries(sample_whisper_asr PRIVATE
  -Wl,--whole-archive houmo_infer -Wl,--no-whole-archive
)
target_include_directories(sample_whisper_asr PRIVATE
  ${HALF_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_INCLUDE}
  ${TOKENIZERS_CPP_PATH}/${LIB_PATH}/include
  ${EIGEN_PATH}
)

# GLM-ASR sample
add_executable(sample_glm_asr sample/sample_glm_asr.cc)
target_link_libraries(sample_glm_asr PRIVATE
  -Wl,--whole-archive houmo_infer -Wl,--no-whole-archive
)
target_include_directories(sample_glm_asr PRIVATE
  ${HALF_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_INCLUDE}
  ${TOKENIZERS_CPP_PATH}/${LIB_PATH}/include
  ${EIGEN_PATH}
)

# Qwen3-ASR sample
add_executable(sample_qwen3_asr sample/sample_qwen3_asr.cc)
target_link_libraries(sample_qwen3_asr PRIVATE
  -Wl,--whole-archive houmo_infer -Wl,--no-whole-archive
)
target_include_directories(sample_qwen3_asr PRIVATE
  ${HALF_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_INCLUDE}
  ${TOKENIZERS_CPP_PATH}/${LIB_PATH}/include
  ${EIGEN_PATH}
)

# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------
if(BUILD_TESTS)
  # Common test include directory
  set(TEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/tests)

  # Tokenizer test
  add_executable(tokenizer_test tests/tokenizer_test.cc)
  target_include_directories(tokenizer_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(tokenizer_test PRIVATE houmo_infer GTest::gtest_main tokenizers_cpp tokenizers_c)
  add_test(NAME TokenizerTest COMMAND tokenizer_test)

  # Embedding test
  add_executable(embedding_test tests/embedding_test.cc)
  target_include_directories(embedding_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(embedding_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME EmbeddingTest COMMAND embedding_test)

  # Sampler test
  add_executable(sampler_test tests/sampler_test.cc)
  target_include_directories(sampler_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(sampler_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME SamplerTest COMMAND sampler_test)

  # Qwen3 LLM test
  add_executable(qwen3_llm_test tests/qwen3_llm_test.cc)
  target_include_directories(qwen3_llm_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(qwen3_llm_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME Qwen3LLMTest COMMAND qwen3_llm_test)

  # Qwen3.5 MLLM test
  add_executable(qwen35_mllm_test tests/qwen35_mllm_test.cc)
  target_include_directories(qwen35_mllm_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(qwen35_mllm_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME Qwen35MLLMTest COMMAND qwen35_mllm_test)

  # Qwen3-VLM test
  add_executable(qwen3_vlm_test tests/qwen3_vlm_test.cc)
  target_include_directories(qwen3_vlm_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(qwen3_vlm_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME Qwen3VLMTest COMMAND qwen3_vlm_test)

  # PerfProfiler test
  add_executable(perf_profiler_test tests/perf_profiler_test.cc)
  target_include_directories(perf_profiler_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(perf_profiler_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME PerfProfilerTest COMMAND perf_profiler_test)

  # AudioProcessor test
  add_executable(audio_processor_test tests/audio_processor_test.cc)
  target_include_directories(audio_processor_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(audio_processor_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME AudioProcessorTest COMMAND audio_processor_test)

  # AudioProcessor 128 mel test (for whisper-large-v3-turbo validation)
  add_executable(audio_processor_128_test tests/audio_processor_128_test.cc)
  target_include_directories(audio_processor_128_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(audio_processor_128_test PRIVATE houmo_infer)
  add_test(NAME AudioProcessor128Test COMMAND audio_processor_128_test)

  add_executable(whisper_test tests/whisper_test.cc)
  target_include_directories(whisper_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(whisper_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME WhisperTest COMMAND whisper_test)

endif()

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS houmo_infer DESTINATION .)
install(TARGETS sample_infer DESTINATION .)
install(TARGETS sample_whisper_asr DESTINATION .)
install(TARGETS sample_glm_asr DESTINATION .)
install(TARGETS sample_qwen3_asr DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/convert_embed.py
        DESTINATION .
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
