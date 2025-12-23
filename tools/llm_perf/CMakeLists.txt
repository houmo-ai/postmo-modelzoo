
cmake_minimum_required(VERSION 3.16.3)
project(llm_perf)

include(CheckCXXCompilerFlag)
if(MSVC)
check_cxx_compiler_flag("/std:c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
else()
  check_cxx_compiler_flag("-std=c++17" SUPPORT_CXX17)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()

include(${CMAKE_SOURCE_DIR}/tcim_runtime.cmake)
include_directories(
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/nlohmann/include
  $ENV{HOUMO_SDK_PATH}/hal/export/include
  $ENV{HOUMO_SDK_PATH}/hal/include
)
link_directories(
  $ENV{HOUMO_SDK_PATH}/hal/lib
)
set(EIGEN_PATH ${CMAKE_SOURCE_DIR}/3rdparty)

if(BACKEND_XH1)
set(SOURCE_FILES HmEmbedding.cc HmvllmInfer.cc HmllmInfer.cc llm_perf.cc)
else()
set(SOURCE_FILES HmEmbedding.cc HmvllmInfer.cc HmllmInfer.cc HmllmInferMultiBatch.cc llm_perf.cc)
endif()

set(INCLUDE_DIRS 
  ${CMAKE_SOURCE_DIR}/include
  ${EIGEN_PATH}
  )

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

add_executable(${PROJECT_NAME}   ${SOURCE_FILES})
target_include_directories(${PROJECT_NAME} PRIVATE ${INCLUDE_DIRS})
if(MSVC)
  target_link_libraries(${PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} ${CMAKE_DL_LIBS} ws2_32 Userenv ntdll Bcrypt kernel32)
else()
  if(ANDROID_ABI)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} dl)
  else()
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} pthread hal_xh2a ${CMAKE_DL_LIBS})
    target_compile_definitions(${PROJECT_NAME} PRIVATE XH2A_HM_SYS)
  endif()
endif()
install(TARGETS ${PROJECT_NAME} DESTINATION .)
