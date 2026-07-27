# Shared TCIM Runtime configuration.

if(_TCIM_RUNTIME_LOADED)
  return()
endif()
set(_TCIM_RUNTIME_LOADED TRUE)

if(NOT DEFINED ENV{TCIM_RUNTIME_PATH} OR "$ENV{TCIM_RUNTIME_PATH}" STREQUAL "")
  message(FATAL_ERROR "TCIM_RUNTIME_PATH environment variable is not set")
endif()

file(TO_CMAKE_PATH "$ENV{TCIM_RUNTIME_PATH}" TCIM_RUNTIME_PATH)
set(TCIM_RUNTIME_INCLUDE "${TCIM_RUNTIME_PATH}/include")
set(TCIM_LIBRARY tcim_runtime_lite)

message(STATUS "TCIM_RUNTIME_PATH: ${TCIM_RUNTIME_PATH}")

include_directories("${TCIM_RUNTIME_INCLUDE}")
link_directories("${TCIM_RUNTIME_PATH}/lib")
