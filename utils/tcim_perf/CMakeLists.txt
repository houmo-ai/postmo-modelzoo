cmake_minimum_required(VERSION 3.8)
project(tcim_perf)

include(../release.cmake)
include_directories($ENV{HOUMO_MODELZOO_PATH}/apis/common/nlohmann/include)

add_executable(${CMAKE_PROJECT_NAME} tcim_perf.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} pthread dl gomp)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)