if(DEFINED ENV{TCIM_RUNTIME_PATH})
    set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
else()
    # Default path for TCIM Runtime
    set(TCIM_RUNTIME_PATH "/opt/venv/houmo/lib/python3.12/site-packages/tcim_lite")
endif()

message(STATUS "TCIM_RUNTIME_PATH: ${TCIM_RUNTIME_PATH}")

if(MSVC)
    set(TCIM_LIBRARY tcim_runtime_lite)
    set(TCIM_RUNTIME_INCLUDE ${TCIM_RUNTIME_PATH}/include)
else()
    set(TCIM_LIBRARY tcim_runtime_lite)
    set(TCIM_RUNTIME_INCLUDE ${TCIM_RUNTIME_PATH}/include)
endif()

include_directories(${TCIM_RUNTIME_INCLUDE})
link_directories(${TCIM_RUNTIME_PATH}/lib)
