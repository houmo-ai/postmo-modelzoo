cmake_minimum_required(VERSION 3.16.3)
project(cosyvoice3-demo VERSION 1.0.0 LANGUAGES CXX)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# C++17 standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include TCIM runtime configuration
include(${CMAKE_CURRENT_SOURCE_DIR}/tcim_runtime.cmake)

# Determine platform-specific library path
if(ANDROID_ABI)
    set(LIB_PATH "Android_xh2")
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    execute_process(
        COMMAND uname -m
        OUTPUT_VARIABLE ARCH
        OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(HOUMO_TARGET "$ENV{HOUMO_TARGET}")
    if(NOT HOUMO_TARGET)
        message(FATAL_ERROR "HOUMO_TARGET environment variable is not set")
    endif()
    set(LIB_PATH "Linux_${ARCH}_${HOUMO_TARGET}")
    message(STATUS "Platform: Linux_${ARCH}_${HOUMO_TARGET}")
else()
    message(WARNING "Unsupported platform: ${CMAKE_SYSTEM_NAME}")
endif()

# Third-party paths
set(TOKENIZER_CPP_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/tokenizers-cpp)
set(AUDIO_HEADER_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/audio/include)
set(HALF_INCLUDE_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include)
set(AUDIO_THIRDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/audio_3rdparty)

# Source files
set(DEMO_SOURCES
    src/demo.cc
    src/hm_audio.cc
    src/hm_embedding.cc
    src/hm_text_frontend.cc
    src/hm_speech_tokenizer.cc
    src/hm_sampling.cc
    src/hm_native_llm.cc
    src/hm_native_flow.cc
    src/hm_native_vocoder.cc
    src/hm_speaker_embedding.cc
    src/hm_cosyvoice3.cc
    src/hm_perf_monitor.cc
    src/miniaudio_impl.cc
)

# Include directories
set(INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty
    $ENV{TCIM_RUNTIME_PATH}/include
    ${TOKENIZER_CPP_PATH}/${LIB_PATH}/include
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/include
    ${HALF_INCLUDE_PATH}
    ${AUDIO_HEADER_PATH}
)

# Library directories
link_directories(
    $ENV{TCIM_RUNTIME_PATH}/lib
    ${TOKENIZER_CPP_PATH}/${LIB_PATH}/lib
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib
)

# Compiler flags

add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")


# Main demo executable
add_executable(cosyvoice3-demo ${DEMO_SOURCES})
target_include_directories(cosyvoice3-demo PRIVATE ${INCLUDE_DIRS})

if(ANDROID_ABI)
    target_link_libraries(cosyvoice3-demo PRIVATE
        tokenizers_cpp
        tokenizers_c
        ${TCIM_LIBRARY}
        dl
        m
        samplerate
        sndfile
        kaldi-native-fbank-core
    )
else()
    target_link_libraries(cosyvoice3-demo PRIVATE
        tokenizers_cpp
        tokenizers_c
        tcim_runtime_lite
        pthread
        ${CMAKE_DL_LIBS}
        m
        samplerate
        sndfile
        kaldi-native-fbank-core
    )
endif()

# Install targets
install(TARGETS cosyvoice3-demo DESTINATION .)
set(AUDIO_LIBS 
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib/libsamplerate.so 
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib/libsndfile.so 
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib/libkaldi-native-fbank-core.so)
install(FILES ${AUDIO_LIBS} DESTINATION .)

if(ANDROID_ABI)
    set(HMM_MODELS ${CMAKE_CURRENT_SOURCE_DIR}/../output)
    set(RAW_MODEL_CONFIGS ${CMAKE_CURRENT_SOURCE_DIR}/../Fun-CosyVoice3-0.5B-2512)
    set(OTHER_FILES 
        ${CMAKE_CURRENT_SOURCE_DIR}/../special_tokens.json
        ${CMAKE_CURRENT_SOURCE_DIR}/../zero_shot_prompt.wav)
    INSTALL(DIRECTORY ${HMM_MODELS}/ DESTINATION ./output)
    INSTALL(DIRECTORY ${RAW_MODEL_CONFIGS}/ DESTINATION ./Fun-CosyVoice3-0.5B-2512)
    INSTALL(FILES ${OTHER_FILES} DESTINATION .)
endif()
# Configuration summary
message(STATUS "")
message(STATUS "CosyVoice3 TTS Demo Configuration:")
message(STATUS "  HOUMO_TARGET: $ENV{HOUMO_TARGET}")
message(STATUS "  TCIM_RUNTIME_PATH: $ENV{TCIM_RUNTIME_PATH}")
message(STATUS "  Platform lib path: ${LIB_PATH}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")