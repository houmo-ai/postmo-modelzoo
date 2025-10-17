
cmake_minimum_required(VERSION 3.16.3)
project(example_cxx_qwen3)

# 在配置阶段执行bash脚本
execute_process(
  COMMAND bash ${CMAKE_SOURCE_DIR}/install.sh
  WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
  RESULT_VARIABLE script_result
)

if(NOT script_result EQUAL 0)
  message(FATAL_ERROR "Bash script failed with result: ${script_result}")
endif()

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

set(TOKENZIER_CPP_PATH ${CMAKE_SOURCE_DIR}/3rdparty/tokenizers-cpp)

set(SOURCE_FILES Hmtokenizer.cc HmQwenInfer.cc qwen3.cc)
set(INCLUDE_DIRS 
  ${TOKENZIER_CPP_PATH}/include 
  ${CMAKE_SOURCE_DIR}/include
  $ENV{HOUMO_PATH}/include)

link_directories($ENV{HOUMO_PATH}/lib
  ${TOKENZIER_CPP_PATH}/lib)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -Wno-deprecated-declarations")


add_executable(${PROJECT_NAME}   ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE pthread tokenizers_cpp tokenizers_c tcim_runtime_lite)

install(TARGETS ${PROJECT_NAME} DESTINATION .)
