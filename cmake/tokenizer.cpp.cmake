cmake_minimum_required(VERSION 3.14)
project(tokenizer LANGUAGES C CXX)

# Include directories
include_directories(include)
if(DEFINED ENV{HOUMO_EXAMPLES_PATH})
    include_directories($ENV{HOUMO_EXAMPLES_PATH}/3rdparty)
else()
    message(FATAL_ERROR "ENV{HOUMO_EXAMPLES_PATH} not set!")
endif()

set(SOURCE_DIR $ENV{HOUMO_EXAMPLES_PATH}/3rdparty/tokenizer.cpp)
include_directories(${SOURCE_DIR}/third_party)

if(MSVC)
    add_compile_options(/utf-8)
    add_definitions(-D_CRT_SECURE_NO_WARNINGS)
endif()

option(UJSON_USE_RAPIDJSON "Use RapidJSON backend" OFF)
if(UJSON_USE_RAPIDJSON)
    add_definitions(-DUJSON_USE_RAPIDJSON)
    include_directories(${SOURCE_DIR}/third_party/rapidjson/include)
endif()

# Oniguruma
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_subdirectory(
    ${SOURCE_DIR}/third_party/oniguruma
    ${CMAKE_CURRENT_BINARY_DIR}/tokenizer_oniguruma
)
include_directories(${SOURCE_DIR}/third_party/oniguruma/src)

# Source files
set(SOURCES
    ${SOURCE_DIR}/src/tokenizer.cpp
    ${SOURCE_DIR}/third_party/utf8proc/utf8proc.c
)

# Library
add_library(tokenizer_lib SHARED ${SOURCES})
target_compile_features(tokenizer_lib PUBLIC cxx_std_11)
if(MSVC)
    set_target_properties(tokenizer_lib PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
endif()
target_include_directories(tokenizer_lib PUBLIC
    ${SOURCE_DIR}/include
    ${SOURCE_DIR}/third_party
    $ENV{HOUMO_EXAMPLES_PATH}/3rdparty
)
target_link_libraries(tokenizer_lib onig)
target_compile_definitions(tokenizer_lib PUBLIC UTF8PROC_STATIC)
