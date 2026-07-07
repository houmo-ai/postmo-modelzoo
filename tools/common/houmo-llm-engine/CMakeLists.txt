cmake_minimum_required(VERSION 3.16.3)
if(POLICY CMP0177)
  cmake_policy(SET CMP0177 NEW)
endif()
project(houmo_infer)

include(CheckCXXCompilerFlag)
include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)

# -----------------------------------------------------------------------------
# Platform configuration
# -----------------------------------------------------------------------------
set(HOUMO_PLATFORM_ANDROID FALSE)
set(HOUMO_PLATFORM_WINDOWS FALSE)
set(HOUMO_PLATFORM_LINUX FALSE)
set(HOUMO_PLATFORM_NAME "unknown")
set(HOUMO_NEED_PTHREAD FALSE)
set(LIB_PATH "")
set(OPENCV_CONFIG_MODE "find_package(OpenCV)")
set(OPENCV_LIB_DIR "")
set(OPENCV_BIN_DIR "")
set(OPENCV_LIBS "")
set(HOUMO_TARGET_LINK_OPTIONS "")
set(HOUMO_INSTALL_AUDIO_DLLS FALSE)
set(HOUMO_INSTALL_AUDIO_SOS FALSE)
set(HOUMO_INSTALL_OPENCV_DLLS FALSE)
set(HOUMO_INSTALL_OPENCV_SOS FALSE)

if(ANDROID_ABI)
  set(HOUMO_PLATFORM_ANDROID TRUE)
  include(${CMAKE_SOURCE_DIR}/cmake/platforms/android.cmake)
elseif(MSVC)
  set(HOUMO_PLATFORM_WINDOWS TRUE)
  include(${CMAKE_SOURCE_DIR}/cmake/platforms/windows.cmake)
elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  set(HOUMO_PLATFORM_LINUX TRUE)
  include(${CMAKE_SOURCE_DIR}/cmake/platforms/linux.cmake)
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
set(OPENCV_REQUIRED_MODULES core imgproc imgcodecs)

if(NOT OpenCV_FOUND OR NOT OpenCV_INCLUDE_DIRS OR NOT OPENCV_LIBS)
  find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_REQUIRED_MODULES})
  set(OPENCV_LIBS ${OpenCV_LIBS})
  set(OPENCV_CONFIG_MODE "find_package(OpenCV)")
endif()

if(NOT OPENCV_LIB_DIR AND OpenCV_FOUND)
  get_target_property(_ocv_loc opencv_core IMPORTED_LOCATION)
  if(NOT _ocv_loc)
    get_target_property(_ocv_loc opencv_core IMPORTED_LOCATION_RELEASE)
  endif()
  if(NOT _ocv_loc)
    get_target_property(_ocv_loc opencv_core IMPORTED_LOCATION_NOCONFIG)
  endif()
  if(_ocv_loc)
    get_filename_component(OPENCV_LIB_DIR "${_ocv_loc}" DIRECTORY)
  endif()
endif()

message(STATUS "OpenCV config mode: ${OPENCV_CONFIG_MODE}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OPENCV_LIBS}")
message(STATUS "OpenCV lib dir: ${OPENCV_LIB_DIR}")
message(STATUS "OpenCV bin dir: ${OPENCV_BIN_DIR}")

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
set(AUDIO_3RDPARTY_RUNTIME_DIR ${AUDIO_3RDPARTY_PATH}/3rdparty_build/bin)

# tokenizer.cpp (HuggingFace tokenizer c++ 3rdparty)
if(DEFINED ENV{HOUMO_EXAMPLES_PATH})
  message(STATUS "Raw env: [$ENV{HOUMO_EXAMPLES_PATH}]")
  set(TOKENIZER_CPP_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/tokenizer.cpp)
  set(HALF_INCLUDE_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include)
else()
  message(FATAL_ERROR "Environment variable HOUMO_EXAMPLES_PATH is not set!...")
endif()
message("TOKENIZER_CPP_PATH: ${TOKENIZER_CPP_PATH}")
add_subdirectory(${TOKENIZER_CPP_PATH} tokenizer_cpp)

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

if(HOUMO_EXPORT_ALL_SYMBOLS)
  set_target_properties(houmo_infer PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
if(HOUMO_TARGET_LINK_OPTIONS)
  target_link_options(houmo_infer PRIVATE ${HOUMO_TARGET_LINK_OPTIONS})
endif()

target_include_directories(houmo_infer PUBLIC
  ${HALF_INCLUDE_PATH}
  ${CMAKE_SOURCE_DIR}/include
  ${TCIM_RUNTIME_INCLUDE}
  ${TOKENIZER_CPP_PATH}/include
  ${TOKENIZER_CPP_PATH}/third_party
  ${EIGEN_PATH}
  ${OpenCV_INCLUDE_DIRS}
  ${AUDIO_3RDPARTY_INCLUDE_DIR}
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/audio/include
)

target_link_directories(houmo_infer PUBLIC
  ${TCIM_RUNTIME_PATH}/lib
  ${AUDIO_3RDPARTY_LIBRARY_DIR}
)

set(HOUMO_INFER_LIBS
  tcim_runtime_lite
  tokenizer_lib
  kaldi-native-fbank-core
  samplerate
  ${CMAKE_DL_LIBS}
)

if(OpenCV_FOUND)
  list(APPEND HOUMO_INFER_LIBS ${OPENCV_LIBS})
  target_compile_definitions(houmo_infer PUBLIC HOUOMO_USE_OPENCV=1)
endif()

if(HOUMO_NEED_PTHREAD)
  list(APPEND HOUMO_INFER_LIBS pthread)
endif()

target_link_libraries(houmo_infer PUBLIC ${HOUMO_INFER_LIBS})

# -----------------------------------------------------------------------------
# Unit tests
# -----------------------------------------------------------------------------
if(BUILD_TESTS)
  set(TEST_INCLUDE_DIR ${CMAKE_SOURCE_DIR}/tests)

  add_executable(tokenizer_test tests/tokenizer_test.cc)
  target_include_directories(tokenizer_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(tokenizer_test PRIVATE houmo_infer GTest::gtest_main tokenizer_lib)
  add_test(NAME TokenizerTest COMMAND tokenizer_test)

  add_executable(embedding_test tests/embedding_test.cc)
  target_include_directories(embedding_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(embedding_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME EmbeddingTest COMMAND embedding_test)

  add_executable(sampler_test tests/sampler_test.cc)
  target_include_directories(sampler_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(sampler_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME SamplerTest COMMAND sampler_test)

  add_executable(perf_profiler_test tests/perf_profiler_test.cc)
  target_include_directories(perf_profiler_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(perf_profiler_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME PerfProfilerTest COMMAND perf_profiler_test)

  add_executable(audio_processor_test tests/audio_processor_test.cc)
  target_include_directories(audio_processor_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(audio_processor_test PRIVATE houmo_infer GTest::gtest_main)
  add_test(NAME AudioProcessorTest COMMAND audio_processor_test)

  add_executable(audio_processor_128_test tests/audio_processor_128_test.cc)
  target_include_directories(audio_processor_128_test PRIVATE ${TEST_INCLUDE_DIR})
  target_link_libraries(audio_processor_128_test PRIVATE houmo_infer)
  add_test(NAME AudioProcessor128Test COMMAND audio_processor_128_test)
endif()

# -----------------------------------------------------------------------------
# Install
# -----------------------------------------------------------------------------
install(TARGETS houmo_infer
  RUNTIME DESTINATION ${LIB_INSTALL_PATH}
  LIBRARY DESTINATION ${LIB_INSTALL_PATH}
  ARCHIVE DESTINATION ${LIB_INSTALL_PATH}
)

install(TARGETS tokenizer_lib
  RUNTIME DESTINATION ${LIB_INSTALL_PATH}
  LIBRARY DESTINATION ${LIB_INSTALL_PATH}
  ARCHIVE DESTINATION ${LIB_INSTALL_PATH}
)

if(HOUMO_INSTALL_AUDIO_DLLS)
  install(DIRECTORY ${AUDIO_3RDPARTY_LIBRARY_DIR}/
    DESTINATION ${LIB_INSTALL_PATH}
    FILES_MATCHING PATTERN "*.dll"
  )
  install(DIRECTORY ${AUDIO_3RDPARTY_RUNTIME_DIR}/
    DESTINATION ${LIB_INSTALL_PATH}
    FILES_MATCHING PATTERN "*.dll"
  )
endif()

if(HOUMO_INSTALL_AUDIO_SOS)
  install(DIRECTORY ${AUDIO_3RDPARTY_LIBRARY_DIR}/
    DESTINATION ${LIB_INSTALL_PATH}
    FILES_MATCHING PATTERN "*.so*"
  )
endif()

if(HOUMO_INSTALL_OPENCV_DLLS AND OPENCV_BIN_DIR)
  file(GLOB _opencv_runtime_files "${OPENCV_BIN_DIR}/opencv_world*.dll")
  list(FILTER _opencv_runtime_files EXCLUDE REGEX "d\\.dll$")
  install(FILES ${_opencv_runtime_files} DESTINATION ${LIB_INSTALL_PATH})
endif()

if(HOUMO_INSTALL_OPENCV_SOS AND OpenCV_FOUND AND NOT OPENCV_LIB_DIR STREQUAL "")
  file(GLOB _opencv_files
    ${OPENCV_LIB_DIR}/libopencv_core.so*
    ${OPENCV_LIB_DIR}/libopencv_imgproc.so*
    ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so*
  )
  list(FILTER _opencv_files EXCLUDE REGEX "\\.a$")
  install(FILES ${_opencv_files} DESTINATION ${LIB_INSTALL_PATH})
endif()

install(FILES ${CMAKE_SOURCE_DIR}/convert_embed.py
        DESTINATION .
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)
