
cmake_minimum_required(VERSION 3.16.3)
project(llm_perf)
include(FetchContent)

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://gitee.com/mirrors/yaml-cpp.git
  GIT_TAG yaml-cpp-0.9.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib" FORCE)
set(YAML_CPP_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static library" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "Disable yaml-cpp install" FORCE)
set(YAML_CPP_DISABLE_FLOAT_IMPLICIT_CONVERSIONS ON CACHE BOOL "Disable float conversions" FORCE)

if(MSVC)
  set(ORIGINAL_CXX_FLAGS ${CMAKE_CXX_FLAGS})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /wd4267 /wd4244")
endif()

FetchContent_MakeAvailable(yaml-cpp)

if(MSVC)
  set(CMAKE_CXX_FLAGS ${ORIGINAL_CXX_FLAGS})
endif()

FetchContent_Declare(
  yaml-cpp
  GIT_REPOSITORY https://gitee.com/mirrors/yaml-cpp.git
  GIT_TAG yaml-cpp-0.9.0
)
set(YAML_CPP_BUILD_TESTS OFF CACHE BOOL "Disable yaml-cpp tests" FORCE)
set(YAML_CPP_BUILD_TOOLS OFF CACHE BOOL "Disable yaml-cpp tools" FORCE)
set(YAML_CPP_BUILD_CONTRIB OFF CACHE BOOL "Disable yaml-cpp contrib" FORCE)
set(YAML_CPP_BUILD_SHARED_LIBS OFF CACHE BOOL "Build static library" FORCE)
set(YAML_CPP_INSTALL OFF CACHE BOOL "Disable yaml-cpp install" FORCE)
FetchContent_MakeAvailable(yaml-cpp)
include(CheckCXXCompilerFlag)
if(MSVC)
  option(ENABLE_ASR "Enable ASR support" OFF)
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEIGEN_MPL2_ONLY /DNOMINMAX")
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /QUIET")
else()
  option(ENABLE_ASR "Enable ASR support" ON)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -DEIGEN_MPL2_ONLY")
endif()

include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)

# houmo-llm-engine paths
set(HOUMO_ENGINE_DIR ${CMAKE_SOURCE_DIR}/../common/houmo-llm-engine)
if(ANDROID_ABI)
  set(_LIB_SUFFIX "Android_xh2")
  set(ENGINE_BUILD_DIR ${HOUMO_ENGINE_DIR}/build_ndk/release)
  set(TOKENIZERS_DIR ${HOUMO_ENGINE_DIR}/3rdparty/tokenizers-cpp/${_LIB_SUFFIX})
  set(AUDIO_3RDPARTY_DIR ${HOUMO_ENGINE_DIR}/3rdparty/audio/build_android)
  set(AUDIO_INC_DIRS
    ${AUDIO_3RDPARTY_DIR}/kaldi-native-fbank/include
  )
  set(AUDIO_LIB_DIRS
    ${AUDIO_3RDPARTY_DIR}/kaldi-native-fbank/lib
    ${AUDIO_3RDPARTY_DIR}/libsamplerate/src
  )
  set(OPENCV_DIR ${HOUMO_ENGINE_DIR}/3rdparty/opencv/${_LIB_SUFFIX})
else()
  execute_process(COMMAND uname -m OUTPUT_VARIABLE _ARCH OUTPUT_STRIP_TRAILING_WHITESPACE)
  set(_LIB_SUFFIX "Linux_${_ARCH}_$ENV{HOUMO_TARGET}")
  set(ENGINE_BUILD_DIR ${HOUMO_ENGINE_DIR}/build)
  set(TOKENIZERS_DIR ${HOUMO_ENGINE_DIR}/3rdparty/tokenizers-cpp/${_LIB_SUFFIX})
  set(AUDIO_3RDPARTY_DIR ${HOUMO_ENGINE_DIR}/3rdparty/audio/3rdparty_build)
  set(AUDIO_INC_DIRS ${AUDIO_3RDPARTY_DIR}/include)
  set(AUDIO_LIB_DIRS ${AUDIO_3RDPARTY_DIR}/lib)
  set(OPENCV_DIR "")
endif()

include_directories(
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/nlohmann/include
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/hpp
  $ENV{HOUMO_SDK_PATH}/hal/export/include
  $ENV{HOUMO_SDK_PATH}/hal/include
  ${HOUMO_ENGINE_DIR}/include
  ${TOKENIZERS_DIR}/include
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/audio/include
  ${CMAKE_SOURCE_DIR}/include
)
if(ENABLE_ASR)
  include_directories(${AUDIO_INC_DIRS})
endif()
link_directories(
  $ENV{HOUMO_SDK_PATH}/hal/lib
  ${ENGINE_BUILD_DIR}
  ${TOKENIZERS_DIR}/lib
)
if(ENABLE_ASR)
  link_directories(${AUDIO_LIB_DIRS})
endif()
set(EIGEN_PATH ${HOUMO_ENGINE_DIR}/3rdparty)

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
  list(APPEND SOURCE_FILES src/asr/HmAsrInfer.cc)
endif()

set(INCLUDE_DIRS
  ${CMAKE_SOURCE_DIR}/include
  ${EIGEN_PATH}
)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")
endif()

add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

add_custom_target(houmo_engine_rebuild ALL
  COMMAND cmake --build ${ENGINE_BUILD_DIR} -j$$(nproc)
  WORKING_DIRECTORY ${HOUMO_ENGINE_DIR}
  COMMENT "Rebuilding houmo-llm-engine libhoumo_infer.a"
)
add_library(houmo_infer STATIC IMPORTED)
set_target_properties(houmo_infer PROPERTIES IMPORTED_LOCATION ${ENGINE_BUILD_DIR}/libhoumo_infer.a)
add_dependencies(houmo_infer houmo_engine_rebuild)
if(MSVC)
  add_definitions(-DYAML_CPP_STATIC_DEFINE)
  if(ENABLE_ASR)
    add_definitions(-DENABLE_ASR)
  endif()
  target_link_libraries(${PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} yaml-cpp  ${CMAKE_DL_LIBS} ws2_32 Userenv ntdll Bcrypt kernel32)
else()
  if(ANDROID_ABI)
    set(ASR_LIBRARIES kaldi-native-fbank-core samplerate)
    if(ENABLE_ASR)
      add_definitions(-DENABLE_ASR)
    else()
      set(ASR_LIBRARIES "")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE
      ${TCIM_LIBRARY} hal_xh2a dl yaml-cpp
      -Wl,--whole-archive houmo_infer -Wl,--no-whole-archive
      tokenizers_cpp tokenizers_c
      ${ASR_LIBRARIES}
      ${OPENCV_DIR}/lib/libopencv_core.so
      ${OPENCV_DIR}/lib/libopencv_imgproc.so
      ${OPENCV_DIR}/lib/libopencv_imgcodecs.so)
  else()
    set(ASR_LIBRARIES kaldi-native-fbank-core samplerate)
    if(ENABLE_ASR)
      add_definitions(-DENABLE_ASR)
    else()
      set(ASR_LIBRARIES "")
    endif()
    target_link_libraries(${PROJECT_NAME} PRIVATE
      ${TCIM_LIBRARY} pthread hal_xh2a ${CMAKE_DL_LIBS} yaml-cpp
      -Wl,--whole-archive houmo_infer -Wl,--no-whole-archive
      tokenizers_cpp tokenizers_c
      ${ASR_LIBRARIES}
      opencv_core opencv_imgproc opencv_imgcodecs)
    target_compile_definitions(${PROJECT_NAME} PRIVATE)
  endif()
endif()
install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/convert_embed.py
        DESTINATION .
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
if(ENABLE_ASR)
  if(ANDROID_ABI)
    message("you should manually copy the following ASR libraries to Aandroid device!")
  else()
    foreach(lib ${AUDIO_LIB_DIRS})
      install(DIRECTORY ${lib}/
              DESTINATION "$ENV{HOUMO_EXAMPLES_PATH}/tools/bin"
              FILES_MATCHING PATTERN "*.so" PATTERN "*.a")
    endforeach()
  endif()
endif()