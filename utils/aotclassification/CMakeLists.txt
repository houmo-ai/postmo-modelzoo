cmake_minimum_required(VERSION 3.8)
project(hdpl_classification)

include($ENV{CMAKE_CONFIG_PATH})

add_executable(${CMAKE_PROJECT_NAME} hdpl_classification.cc)
target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} ${IDNNL_LIBRARY} ${HDPL_LIBRARY} pthread dl gomp)
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)
