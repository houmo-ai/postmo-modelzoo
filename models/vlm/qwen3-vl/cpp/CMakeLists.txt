
cmake_minimum_required(VERSION 3.16.3)
project(example_cxx_qwen3_vl)

include(CheckCXXCompilerFlag)
if(MSVC)
check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /DEIGEN_MPL2_ONLY")
else()
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -DEIGEN_MPL2_ONLY")
endif()

include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)
set(TOKENZIER_CPP_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tokenizers-cpp)
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/3rdparty)

# OpenCV modules used by current cpp code:
# - core (Mat/split/convertTo)
# - imgproc (resize/cvtColor)
# - imgcodecs (imread)
set(OPENCV_REQUIRED_MODULES core imgproc imgcodecs)

if(ANDROID_ABI)
  # Android cross-compile: allow explicit OpenCV include/lib paths.
  # Required env vars for explicit mode:
  #   OPENCV_INCLUDE_DIRS
  #   OPENCV_LIB_DIR (directory containing libopencv_*.so for target ABI)
  set(OpenCV_INCLUDE_DIRS "$ENV{OPENCV_INCLUDE_DIRS}")
  set(OPENCV_LIB_DIR "$ENV{OPENCV_LIB_DIR}")

  if(NOT OpenCV_INCLUDE_DIRS STREQUAL "" AND NOT OPENCV_LIB_DIR STREQUAL "")
    set(OPENCV_LIBS
      ${OPENCV_LIB_DIR}/libopencv_core.so
      ${OPENCV_LIB_DIR}/libopencv_imgproc.so
      ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
    )
    message(STATUS "OpenCV(Android) include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV(Android) libs: ${OPENCV_LIBS}")
  else()
    message(STATUS "OPENCV_INCLUDE_DIRS/OPENCV_LIB_DIR not set, fallback to find_package(OpenCV)")
    find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_REQUIRED_MODULES})
    set(OPENCV_LIBS ${OpenCV_LIBS})
    message(STATUS "OpenCV found: ${OpenCV_VERSION}")
    message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV libs: ${OPENCV_LIBS}")
  endif()
else()
  find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_REQUIRED_MODULES})
  set(OPENCV_LIBS ${OpenCV_LIBS})
  message(STATUS "OpenCV found: ${OpenCV_VERSION}")
  message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
  message(STATUS "OpenCV libs: ${OPENCV_LIBS}")
endif()



if(MSVC)
  set(LIB_PATH "win11_x64")
else()
  if(NOT ANDROID_ABI)
    set(SYS_NAME "")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(SYS_NAME "Linux")
    else()
      message(WARNING "Current System is ${CMAKE_SYSTEM_NAME}, Unsupported System!")
    endif()
    execute_process(
      COMMAND uname -m
      OUTPUT_VARIABLE ARCH
      OUTPUT_STRIP_TRAILING_WHITESPACE
    )
    set(HOUMO_TARGET "$ENV{HOUMO_TARGET}")
    set(LIB_PATH "${SYS_NAME}_${ARCH}_${HOUMO_TARGET}")
    message(STATUS "LIB_PATH: ${LIB_PATH}")
  else()
    set(LIB_PATH "Android_xh2")
  endif()
endif()


set(SOURCE_FILES
    src/HmImageProcessor.cc
    src/HmQwenVLTokenizer.cc
    src/HmQwenVLInfer.cc
    src/demo.cc
)

# Half precision floating point library path
if(DEFINED ENV{HALF_INCLUDE_PATH})
    set(HALF_INCLUDE_PATH $ENV{HALF_INCLUDE_PATH})
elseif(DEFINED ENV{HOUMO_EXAMPLES_PATH})
    set(HALF_INCLUDE_PATH $ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include)
else()
    message(WARNING "HALF_INCLUDE_PATH not set, using default path")
    set(HALF_INCLUDE_PATH /usr/include/half)
endif()

set(INCLUDE_DIRS
  ${HALF_INCLUDE_PATH}
  ${TOKENZIER_CPP_PATH}/${LIB_PATH}/include
  ${CMAKE_SOURCE_DIR}/include
  $ENV{HOUMO_PATH}/include
  ${EIGEN_PATH}
  ${OpenCV_INCLUDE_DIRS}
)

link_directories($ENV{HOUMO_PATH}/lib
  ${TOKENZIER_CPP_PATH}/${LIB_PATH}/lib)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()


add_executable(${PROJECT_NAME} ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})

if(MSVC)
  target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp tokenizers_c ${TCIM_LIBRARY} ${CMAKE_DL_LIBS} ws2_32 Userenv ntdll Bcrypt kernel32 ${OPENCV_LIBS})
else()
  if(ANDROID_ABI)
    target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp tokenizers_c ${TCIM_LIBRARY} dl ${OPENCV_LIBS})
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp tokenizers_c tcim_runtime_lite pthread ${CMAKE_DL_LIBS} ${OPENCV_LIBS})
  endif()
endif()

install(TARGETS ${PROJECT_NAME} DESTINATION .)
