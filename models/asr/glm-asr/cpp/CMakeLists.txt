cmake_minimum_required(VERSION 3.16.3)
project(demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/utf-8 /wd4996)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
  add_compile_options(-Wno-deprecated-declarations)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_MPL2_ONLY")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")
endif()

# ---------------------------------------------------------------------
# Load houmo_engine config (houmo_engine INTERFACE library)
# ---------------------------------------------------------------------
include("$ENV{HOUMO_EXAMPLES_PATH}/cmake/houmo_engine.cmake")

# ---------------------------------------------------------------------
# Sample target
# ---------------------------------------------------------------------
add_executable(demo
  demo.cc
  src/glm_asr_model.cc
)

target_include_directories(demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(demo PRIVATE houmo_engine)
install(TARGETS demo DESTINATION .)
