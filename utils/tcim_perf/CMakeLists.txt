cmake_minimum_required(VERSION 3.8)
project(tcim_perf)

include($ENV{HOUMO_MODELZOO_PATH}/utils/release.cmake)

add_executable(${CMAKE_PROJECT_NAME} tcim_perf.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} pthread dl gomp)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)