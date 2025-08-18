cmake_minimum_required(VERSION 3.16.3)
project(example_module_pool)

set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
set(HOUMO_EXAMPLES_PATH $ENV{HOUMO_EXAMPLES_PATH})

find_library(TCIM_LIBRARY NAMES tcim_runtime_lite PATHS ${TCIM_RUNTIME_PATH}/lib NO_DEFAULT_PATH)
find_package(OpenCV REQUIRED)

include_directories(
	${TCIM_RUNTIME_PATH}/include
	${HOUMO_EXAMPLES_PATH}/common/hpp
	${OpenCV_INCLUDE_DIRS}
)
message("OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
message("TCIM_RUNTIME_PATH=${TCIM_RUNTIME_PATH}")
message("HOUMO_EXAMPLES_PATH=${HOUMO_EXAMPLES_PATH}")

if(MSVC)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(${CMAKE_PROJECT_NAME} module_pool_example.cc)

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY})

if(MSVC)
else()
	target_link_libraries(${CMAKE_PROJECT_NAME} pthread dl gomp stdc++fs)
endif()

install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)
