cmake_minimum_required(VERSION 3.15)

project(yaml-cpp LANGUAGES CXX)

file(GLOB yaml-cpp-sources CONFIGURE_DEPENDS "src/*.cpp")

add_library(yaml-cpp STATIC ${yaml-cpp-sources})
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)

target_include_directories(yaml-cpp
  PUBLIC
    ${PROJECT_SOURCE_DIR}/include
  PRIVATE
    ${PROJECT_SOURCE_DIR}/src
)

target_compile_options(yaml-cpp
  PRIVATE
    $<$<CXX_COMPILER_ID:MSVC>:/W3 /wd4127 /wd4355>
)

target_compile_definitions(yaml-cpp
  PUBLIC
    YAML_CPP_STATIC_DEFINE
  PRIVATE
    YAML_CPP_NO_CONTRIB
)

set_target_properties(yaml-cpp PROPERTIES
  CXX_STANDARD 11
  CXX_STANDARD_REQUIRED ON
  POSITION_INDEPENDENT_CODE ON
)
