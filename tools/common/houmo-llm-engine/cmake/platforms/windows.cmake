# Windows platform configuration for houmo_infer.

set(HOUMO_PLATFORM_NAME "windows")
set(HOUMO_NEED_PTHREAD FALSE)
set(HOUMO_CXX17_FLAG "/std:c++17")
set(HOUMO_COMPILE_OPTIONS /utf-8 /wd4996)
set(HOUMO_RELEASE_FLAGS "/O2 /DNDEBUG")
set(HOUMO_DEBUG_FLAGS "")
set(HOUMO_EXPORT_ALL_SYMBOLS TRUE)
set(HOUMO_TARGET_LINK_OPTIONS /IGNORE:4197)

if(DEFINED ENV{HOUMO_EXAMPLES_PATH})
  file(TO_CMAKE_PATH "$ENV{HOUMO_EXAMPLES_PATH}" HOUMO_EXAMPLES_ROOT)
  set(LIB_INSTALL_PATH "${HOUMO_EXAMPLES_ROOT}/tools/common/lib")
endif()

if(DEFINED ENV{OPENCV_PATH})
  message(STATUS "Use Windows prebuilt OpenCV")
  file(TO_CMAKE_PATH "$ENV{OPENCV_PATH}" OPENCV_ROOT)
  set(OPENCV_PATH "${OPENCV_ROOT}/build")
  set(OpenCV_INCLUDE_DIRS "${OPENCV_PATH}/include")
  set(OPENCV_LIB_DIR "${OPENCV_PATH}/x64/vc16/lib")
  set(OPENCV_BIN_DIR "${OPENCV_PATH}/x64/vc16/bin")

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

set(HOUMO_INSTALL_AUDIO_DLLS TRUE)
set(HOUMO_INSTALL_AUDIO_SOS FALSE)
set(HOUMO_INSTALL_OPENCV_DLLS TRUE)
set(HOUMO_INSTALL_OPENCV_SOS FALSE)
