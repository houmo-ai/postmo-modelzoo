cmake_minimum_required(VERSION 3.8)
project(tcim_perf)

include($ENV{HOUMO_EXAMPLES_PATH}/tools/tcim_runtime.cmake)
include_directories($ENV{HOUMO_EXAMPLES_PATH}/apis/common/nlohmann/include)

add_executable(${CMAKE_PROJECT_NAME} tcim_perf.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} dl)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)