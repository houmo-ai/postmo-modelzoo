cmake_minimum_required(VERSION 3.14)
project(tokenizer LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)
include_directories(third_party)
if(DEFINED ENV{HOUMO_EXAMPLES_PATH})
    include_directories($ENV{HOUMO_EXAMPLES_PATH}/apis/common/nlohmann/include)
else()
    include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../nlohmann/include)
endif()

if(MSVC)
    add_compile_options(/utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

option(UJSON_USE_RAPIDJSON "Use RapidJSON backend" OFF)
if(UJSON_USE_RAPIDJSON)
    add_definitions(-DUJSON_USE_RAPIDJSON)
    include_directories(third_party/rapidjson/include)
endif()

# Oniguruma
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(third_party/oniguruma)
include_directories(third_party/oniguruma/src)

# Source files
set(SOURCES
    src/tokenizer.cpp
    third_party/utf8proc/utf8proc.c
)

# Library
add_library(tokenizer_lib SHARED ${SOURCES})
target_include_directories(tokenizer_lib PUBLIC include third_party ${CMAKE_CURRENT_SOURCE_DIR}/../nlohmann/include)
target_link_libraries(tokenizer_lib onig)
target_compile_definitions(tokenizer_lib PUBLIC UTF8PROC_STATIC)
