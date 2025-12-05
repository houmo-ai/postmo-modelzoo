# build definitions

if(MSVC)
  add_definitions(/W1)
  add_compile_options("/utf-8")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /O2")
else()
  add_definitions(-w)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")
endif()

# include paths
include_directories($ENV{TCIM_RUNTIME_PATH}/include)

# lib paths
link_directories($ENV{TCIM_RUNTIME_PATH}/lib)
set(TCIM_LIBRARY "tcim_runtime_lite")
