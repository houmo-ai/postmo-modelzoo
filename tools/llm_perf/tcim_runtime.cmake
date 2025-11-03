# build definitions
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

# include paths
include_directories($ENV{TCIM_RUNTIME_PATH}/include)

# lib paths
link_directories($ENV{TCIM_RUNTIME_PATH}/lib)
set(TCIM_LIBRARY "-ltcim_runtime_lite")
