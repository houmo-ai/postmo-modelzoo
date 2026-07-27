cmake_minimum_required(VERSION 3.16.3)
project(example_resnet50_pipeline LANGUAGES CXX)

# Set RPATH so the executable finds .so in its own directory at runtime
set(CMAKE_INSTALL_RPATH "$ORIGIN")
set(CMAKE_BUILD_RPATH "$ORIGIN")
set(CMAKE_SKIP_BUILD_RPATH FALSE)
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

set(TCIM_RUNTIME_PATH $ENV{TCIM_RUNTIME_PATH})
set(HOUMO_EXAMPLES_PATH $ENV{HOUMO_EXAMPLES_PATH})

include_directories(${TCIM_RUNTIME_PATH}/include)
include_directories(${HOUMO_EXAMPLES_PATH}/utils/cpp/include)
include_directories(${HOUMO_EXAMPLES_PATH}/3rdparty)
link_directories(${TCIM_RUNTIME_PATH}/lib)

# OpenCV configuration
if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  # OHOS prebuilt OpenCV
  if(DEFINED ENV{HOUMO_TARGET})
    set(OPENCV_PREBUILT_PATH ${CMAKE_SOURCE_DIR}/../../3rdparty/opencv/OHOS_aarch64_$ENV{HOUMO_TARGET})
  endif()
  if(EXISTS ${OPENCV_PREBUILT_PATH}/include)
    include_directories(${OPENCV_PREBUILT_PATH}/include)
    set(OpenCV_LIBS
      ${OPENCV_PREBUILT_PATH}/lib/libopencv_core.so
      ${OPENCV_PREBUILT_PATH}/lib/libopencv_imgproc.so
      ${OPENCV_PREBUILT_PATH}/lib/libopencv_imgcodecs.so
    )
    message(STATUS "OHOS OpenCV: ${OPENCV_PREBUILT_PATH}")
  else()
    message(WARNING "OHOS OpenCV not found at ${OPENCV_PREBUILT_PATH}")
  endif()
else()
  find_package(OpenCV REQUIRED)
  include_directories(${OpenCV_INCLUDE_DIRS})
  message("OpenCV_INCLUDE_DIRS=${OpenCV_INCLUDE_DIRS}")
endif()

message("TCIM_RUNTIME_PATH=${TCIM_RUNTIME_PATH}")
message("HOUMO_EXAMPLES_PATH=${HOUMO_EXAMPLES_PATH}")

if(MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17")
  add_compile_options(/utf-8)
else()
  add_definitions(-w)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17")
endif()
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2")

add_executable(${CMAKE_PROJECT_NAME} resnet50_pipeline.cc)

target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS})

if(MSVC)
  target_link_libraries(${CMAKE_PROJECT_NAME} tcim_runtime_lite)
elseif(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  target_link_libraries(${CMAKE_PROJECT_NAME} tcim_runtime_lite pthread dl c++ c++abi)
else()
  target_link_libraries(${CMAKE_PROJECT_NAME} tcim_runtime_lite pthread dl gomp stdc++fs)
endif()

install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)

# Install runtime shared libraries
if(CMAKE_SYSTEM_NAME STREQUAL "OHOS")
  install(FILES "${TCIM_RUNTIME_PATH}/lib/libtcim_runtime_lite.so" DESTINATION .)
  if(OPENCV_PREBUILT_PATH)
    install(FILES
      "${OPENCV_PREBUILT_PATH}/lib/libopencv_core.so"
      "${OPENCV_PREBUILT_PATH}/lib/libopencv_imgproc.so"
      "${OPENCV_PREBUILT_PATH}/lib/libopencv_imgcodecs.so"
      DESTINATION .)
  endif()
else()
  install(FILES "${TCIM_RUNTIME_PATH}/lib/libtcim_runtime_lite.so" DESTINATION .)
endif()
