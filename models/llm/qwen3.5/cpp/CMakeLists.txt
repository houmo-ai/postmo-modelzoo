cmake_minimum_required(VERSION 3.16.3)
project(qwen35_mllm_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

if(MSVC)
  add_compile_options(/utf-8 /wd4996)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /O2 /DNDEBUG")
else()
  add_compile_options(-Wno-deprecated-declarations)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -DEIGEN_MPL2_ONLY")
  set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -ggdb")
endif()

set(OPENCV_REQUIRED_MODULES core imgproc imgcodecs)
set(OPENCV_CONFIG_MODE "find_package(OpenCV)")
set(OPENCV_LIB_DIR "")
set(OPENCV_LIBS "")

if(ANDROID_ABI)
  set(OPENCV_PATH ${HM_LLM_ENGINE_DIR}/3rdparty/opencv/Android_xh2)
  set(OpenCV_INCLUDE_DIRS ${OPENCV_PATH}/include)
  set(OPENCV_LIB_DIR ${OPENCV_PATH}/lib)
  set(OPENCV_LIBS
    ${OPENCV_LIB_DIR}/libopencv_core.so
    ${OPENCV_LIB_DIR}/libopencv_imgproc.so
    ${OPENCV_LIB_DIR}/libopencv_imgcodecs.so
  )
  set(OpenCV_FOUND TRUE)
  set(OPENCV_CONFIG_MODE "Android prebuilt OpenCV")
elseif(MSVC AND DEFINED ENV{OPENCV_PATH})
  file(TO_CMAKE_PATH "$ENV{OPENCV_PATH}" OPENCV_ROOT)
  set(OPENCV_PATH "${OPENCV_ROOT}/build")
  set(OpenCV_INCLUDE_DIRS "${OPENCV_PATH}/include")
  set(OPENCV_LIB_DIR "${OPENCV_PATH}/x64/vc16/lib")

  file(GLOB _opencv_world_release "${OPENCV_LIB_DIR}/opencv_world*.lib")
  list(FILTER _opencv_world_release EXCLUDE REGEX "d\\.lib$")
  if(_opencv_world_release)
    set(OPENCV_LIBS ${_opencv_world_release})
    set(OpenCV_FOUND TRUE)
    set(OPENCV_CONFIG_MODE "Windows prebuilt OpenCV world")
  else()
    file(GLOB OPENCV_LIBS
      "${OPENCV_LIB_DIR}/opencv_core*.lib"
      "${OPENCV_LIB_DIR}/opencv_imgproc*.lib"
      "${OPENCV_LIB_DIR}/opencv_imgcodecs*.lib"
    )
    list(FILTER OPENCV_LIBS EXCLUDE REGEX "d\\.lib$")
    if(OPENCV_LIBS)
      set(OpenCV_FOUND TRUE)
      set(OPENCV_CONFIG_MODE "Windows prebuilt OpenCV modules")
    endif()
  endif()
endif()

if(NOT OpenCV_FOUND OR NOT OpenCV_INCLUDE_DIRS OR NOT OPENCV_LIBS)
  find_package(OpenCV REQUIRED COMPONENTS ${OPENCV_REQUIRED_MODULES})
  set(OPENCV_LIBS ${OpenCV_LIBS})
  set(OPENCV_CONFIG_MODE "find_package(OpenCV)")
endif()

message(STATUS "OpenCV config mode: ${OPENCV_CONFIG_MODE}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libs: ${OPENCV_LIBS}")

include(${CMAKE_CURRENT_SOURCE_DIR}/hm_engine.cmake)

add_executable(demo
  demo.cc
  src/qwen35_mllm_model.cc
)

target_include_directories(demo PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(demo PRIVATE hm_engine ${OPENCV_LIBS})
install(TARGETS demo DESTINATION .)
