# build definitions
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

# include paths
include_directories($ENV{TCIM_RUNTIME_PATH}/include)
include_directories($ENV{HOUMO_EXAMPLES_PATH}/tools/common)
if(MSVC)
include_directories($ENV{HOUMO_EXAMPLES_PATH}/tools/common/include)
endif()
# lib paths
link_directories($ENV{TCIM_RUNTIME_PATH}/lib)
link_directories($ENV{HOUMO_SDK_PATH}/hal/lib)
set(TCIM_LIBRARY "tcim_runtime_lite")
