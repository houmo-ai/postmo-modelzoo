cmake_minimum_required(VERSION 3.16.3)
project(dev_monitor)

message("HOUMO_SDK_PATH=$ENV{HOUMO_SDK_PATH}")
include_directories(
	$ENV{HOUMO_SDK_PATH}/hal/export/include
	$ENV{HOUMO_SDK_PATH}/hal/include
	$ENV{HOUMO_SDK_PATH}/include
)
link_directories($ENV{HOUMO_SDK_PATH}/hal/lib)

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(${CMAKE_PROJECT_NAME} monitor_device_mem.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} hal_xh2a)
# install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)
