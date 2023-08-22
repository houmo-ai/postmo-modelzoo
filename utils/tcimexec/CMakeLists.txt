cmake_minimum_required(VERSION 3.8)
project(tcimexec)

set(TCIM_PATH $ENV{TCIM_PATH})
set(HOUMO_PATH $ENV{HOUMO_PATH})

set(TCIM_INC_PATH $ENV{TCIM_INC_PATH})
set(TCIM_LIB_PATH $ENV{TCIM_LIB_PATH})
set(HDPL_INC_PATH $ENV{HDPL_INC_PATH})
set(HDPL_LIB_PATH $ENV{HDPL_LIB_PATH})

# build definitions
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

# include paths
include_directories(${TCIM_INC_PATH})
include_directories(${HDPL_INC_PATH})

# lib paths
find_library(TCIM_LIBRARY NAMES tvm PATHS ${TCIM_LIB_PATH} NO_DEFAULT_PATH)
find_library(HDPL_LIBRARY NAMES hdplrt PATHS ${HDPL_LIB_PATH} NO_DEFAULT_PATH)
message("TCIM_LIBRARY is ${TCIM_LIBRARY}")
message("HDPL_LIBRARY is ${HDPL_LIBRARY}")

add_executable(${CMAKE_PROJECT_NAME} main.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} ${HDPL_LIBRARY} pthread dl gomp)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)
