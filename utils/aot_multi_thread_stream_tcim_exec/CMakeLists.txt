cmake_minimum_required(VERSION 3.8)
project(multi_thread_stream_tcimexec)

include($ENV{CMAKE_CONFIG_PATH})

add_executable(${CMAKE_PROJECT_NAME} main.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} ${IDNNL_LIBRARY} ${HDPL_LIBRARY} pthread dl gomp)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)