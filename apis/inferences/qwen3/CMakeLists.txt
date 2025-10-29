
cmake_minimum_required(VERSION 3.16.3)
project(example_cxx_qwen3)

include(CheckCXXCompilerFlag)
if(NOT MSVC)
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "-std=c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
else()
  check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "/std:c++17 ${CMAKE_CXX_FLAGS}")
  set(CMAKE_CUDA_STANDARD 17)
endif()

if(ANDROID_ABI)
  include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)
endif()

set(TOKENZIER_CPP_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tokenizers-cpp)
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/3rdparty)
if(NOT MSVC)
  if(NOT ANDROID_ABI)
    set(SYS_NAME "")
    if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(SYS_NAME "Linux")  # 确认是 Linux 系统，设置为 "Linux"
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


set(SOURCE_FILES Hmtokenizer.cc HmQwenInfer.cc qwen3.cc)
set(INCLUDE_DIRS 
  ${TOKENZIER_CPP_PATH}/${LIB_PATH}/include 
  ${CMAKE_SOURCE_DIR}/include
  $ENV{HOUMO_PATH}/include
  ${EIGEN_PATH}
  )

link_directories($ENV{HOUMO_PATH}/lib
  ${TOKENZIER_CPP_PATH}/${LIB_PATH}/lib)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wno-deprecated-declarations")


add_executable(${PROJECT_NAME}   ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
if(NOT MSVC)
  if(ANDROID_ABI)
    target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp tokenizers_c ${TCIM_LIBRARY} dl)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE tokenizers_cpp tokenizers_c tcim_runtime_lite pthread  ${CMAKE_DL_LIBS})
  endif()
endif()
install(TARGETS ${PROJECT_NAME} DESTINATION .)
