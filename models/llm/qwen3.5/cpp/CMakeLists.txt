cmake_minimum_required(VERSION 3.16.3)
project(qwen35_mllm_demo)

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

include(${CMAKE_CURRENT_SOURCE_DIR}/hm_engine.cmake)

set(OPENCV_INCLUDE_DIR ${HM_LLM_ENGINE_DIR}/3rdparty/opencv/Android_xh2/include)

add_executable(demo
  demo.cc
  src/qwen35_mllm_model.cc
)

target_include_directories(demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OPENCV_INCLUDE_DIR}
)

target_link_libraries(demo PRIVATE hm_engine)
install(TARGETS demo DESTINATION .)
