# tcim runtime cmake configuration
# This file is used to find and configure tcim_runtime library

if(DEFINED ENV{TCIM_RUNTIME_PATH})
    set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
else()
    set(TCIM_RUNTIME_PATH "${CMAKE_SOURCE_DIR}/../../tcim_runtime")
endif()

if(MSVC)
    set(TCIM_LIBRARY tcim_runtime_lite)
    set(TCIM_RUNTIME_INCLUDE ${TCIM_RUNTIME_PATH}/include)
else()
    if(ANDROID_ABI)
        set(TCIM_LIBRARY tcim_runtime_lite)
        set(TCIM_RUNTIME_INCLUDE ${TCIM_RUNTIME_PATH}/include)
    else()
        set(TCIM_LIBRARY tcim_runtime_lite)
        set(TCIM_RUNTIME_INCLUDE ${TCIM_RUNTIME_PATH}/include)
    endif()
endif()

include_directories(${TCIM_RUNTIME_INCLUDE})
link_directories(${TCIM_RUNTIME_PATH}/lib)
