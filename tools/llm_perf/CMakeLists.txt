
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
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DEIGEN_MPL2_ONLY /DNOMINMAX")
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} /QUIET")
else()
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -DEIGEN_MPL2_ONLY")
endif()

include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)
include_directories(
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/nlohmann/include
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/hpp
  $ENV{HOUMO_SDK_PATH}/hal/export/include
  $ENV{HOUMO_SDK_PATH}/hal/include
  ${CMAKE_SOURCE_DIR}
)
link_directories(
  $ENV{HOUMO_SDK_PATH}/hal/lib
)
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/3rdparty)

set(SOURCE_FILES
  perf_dumper/perf_dumper.cc
  host_monitor/host_monitor.cc
  device_monitor/device_monitor.cc
  perf_tracker/inference_perf_tracker.cc
  HmEmbedding.cc
  HmvllmInfer.cc
  HmllmInfer.cc
  HmllmInferMultiBatch.cc
  llm_perf.cc
)

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

add_executable(${PROJECT_NAME}   ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
if(MSVC)
  add_definitions(-DYAML_CPP_STATIC_DEFINE)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} yaml-cpp  ${CMAKE_DL_LIBS} ws2_32 Userenv ntdll Bcrypt kernel32)
else()
  if(ANDROID_ABI)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} hal_xh2a dl yaml-cpp)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} pthread hal_xh2a ${CMAKE_DL_LIBS} yaml-cpp)
    target_compile_definitions(${PROJECT_NAME} PRIVATE)
  endif()
endif()
install(TARGETS ${PROJECT_NAME} DESTINATION .)
install(FILES ${CMAKE_SOURCE_DIR}/convert_embed.py
        DESTINATION .
        PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                    GROUP_READ GROUP_EXECUTE
                    WORLD_READ WORLD_EXECUTE)