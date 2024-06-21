set(TCIM_PATH $ENV{TCIM_PATH})
set(HOUMO_PATH $ENV{HOUMO_PATH})

set(TCIM_INC_PATH $ENV{TCIM_INC_PATH})
set(TCIM_LIB_PATH $ENV{TCIM_LIB_PATH})
set(HDPL_INC_PATH $ENV{HDPL_INC_PATH})
set(HDPL_LIB_PATH $ENV{HDPL_LIB_PATH})

# build definitions
add_definitions(-w)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O2")

# include paths
include_directories(${TCIM_INC_PATH})
include_directories(${HDPL_INC_PATH})
include_directories($ENV{MODELZOO_PATH}/utils/common)

# lib paths
find_library(TCIM_LIBRARY NAMES tvm_runtime PATHS ${TCIM_LIB_PATH} NO_DEFAULT_PATH)
find_library(HDPL_LIBRARY NAMES hdplrt PATHS ${HDPL_LIB_PATH} NO_DEFAULT_PATH)
find_library(IDNNL_LIBRARY NAMES idnnl PATHS ${HDPL_LIB_PATH} NO_DEFAULT_PATH)
message("TCIM_LIBRARY is ${TCIM_LIBRARY}")
message("HDPL_LIBRARY is ${HDPL_LIBRARY}")
message("IDNNL_LIBRARY is ${IDNNL_LIBRARY}")