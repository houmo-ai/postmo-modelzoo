cmake_minimum_required(VERSION 3.16.3)
project(qwen3_tts_text_demo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/utf-8 /wd4996)
else()
  add_compile_options(-Wno-deprecated-declarations)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_MPL2_ONLY")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")
endif()

include("$ENV{HOUMO_EXAMPLES_PATH}/cmake/houmo_engine.cmake")

if(ANDROID_ABI)
  if(NOT ANDROID_ABI STREQUAL "arm64-v8a")
    message(FATAL_ERROR "Qwen3-TTS Android build only supports arm64-v8a")
  endif()
  add_compile_definitions(QWEN3_TTS_ANDROID=1)
endif()

add_library(qwen3_tts_streaming_frontend STATIC
  src/qwen3_tts_code_predictor.cc
  src/qwen3_tts_code_predictor_embedding.cc
  src/qwen3_tts_codec_embedding.cc
  src/qwen3_tts_prefill_decode_runtime.cc
  src/qwen3_tts_sampler.cc
  src/qwen3_tts_streaming_prompt_builder.cc
  src/qwen3_tts_streaming_generator.cc
  src/qwen3_tts_talker.cc
  src/qwen3_tts_stateful_decoder.cc
  src/qwen3_tts_wav_writer.cc
  src/qwen3_tts_text_embedding.cc
  src/qwen3_tts_text_processor.cc
  src/qwen3_tts_text_projection.cc
)

target_include_directories(qwen3_tts_streaming_frontend PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(qwen3_tts_streaming_frontend PUBLIC houmo_engine)

if(ANDROID_ABI)
  target_link_libraries(qwen3_tts_streaming_frontend PUBLIC dl m)
endif()

add_executable(qwen3_tts_text_demo demo.cc)
target_link_libraries(qwen3_tts_text_demo PRIVATE qwen3_tts_streaming_frontend)
install(TARGETS qwen3_tts_text_demo DESTINATION .)

if(ANDROID_ABI)
  set(QWEN3_TTS_MODEL_DIRS
      "${CMAKE_CURRENT_SOURCE_DIR}/../Qwen3-TTS-12Hz-0.6B-CustomVoice"
      "${CMAKE_CURRENT_SOURCE_DIR}/../Qwen3-TTS-12Hz-1.7B-CustomVoice")
  set(QWEN3_TTS_OUTPUT_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../output")

  foreach(QWEN3_TTS_MODEL_DIR IN LISTS QWEN3_TTS_MODEL_DIRS)
    get_filename_component(QWEN3_TTS_MODEL_NAME "${QWEN3_TTS_MODEL_DIR}" NAME)
    if(EXISTS "${QWEN3_TTS_MODEL_DIR}")
      install(DIRECTORY "${QWEN3_TTS_MODEL_DIR}/"
              DESTINATION "${QWEN3_TTS_MODEL_NAME}")
    else()
      message(
        WARNING "Tokenizer/model directory not found: ${QWEN3_TTS_MODEL_DIR}")
    endif()
  endforeach()

  if(EXISTS "${QWEN3_TTS_OUTPUT_DIR}")
    install(DIRECTORY "${QWEN3_TTS_OUTPUT_DIR}/" DESTINATION output)
  else()
    message(WARNING "Compiled model output directory not found: ${QWEN3_TTS_OUTPUT_DIR}")
  endif()
endif()
