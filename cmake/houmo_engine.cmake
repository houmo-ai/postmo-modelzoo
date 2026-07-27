# Reusable CMake module for utils/cpp/houmo_engine.
#
# Usage:
#   include($ENV{HOUMO_EXAMPLES_PATH}/cmake/houmo_engine.cmake)
#   target_link_libraries(your_target PRIVATE houmo_engine)
#
# Sets up:
#   - houmo_engine INTERFACE library (includes + link dirs + link libs)
#   - houmo_infer_build custom target on Linux/Android when auto-build is enabled

if(_HM_ENGINE_LOADED)
  return()
endif()
set(_HM_ENGINE_LOADED TRUE)

if(NOT DEFINED ENV{HOUMO_EXAMPLES_PATH})
  message(FATAL_ERROR "HOUMO_EXAMPLES_PATH not set")
endif()

file(TO_CMAKE_PATH "$ENV{HOUMO_EXAMPLES_PATH}" HOUMO_EXAMPLES_PATH_CMAKE)
include("${HOUMO_EXAMPLES_PATH_CMAKE}/cmake/tcim_runtime.cmake")

set(HOUMO_ENGINE_DIR "${HOUMO_EXAMPLES_PATH_CMAKE}/utils/cpp/houmo_engine")
if(ANDROID_ABI)
  set(HM_LIB_DIR "${HOUMO_EXAMPLES_PATH_CMAKE}/utils/android")
else()
  set(HM_LIB_DIR "${HOUMO_EXAMPLES_PATH_CMAKE}/utils/lib")
endif()

set(HM_TOKENIZER_INC "${HOUMO_EXAMPLES_PATH_CMAKE}/3rdparty/tokenizer.cpp/include")
set(HM_THIRDPARTY_INC "${HOUMO_EXAMPLES_PATH_CMAKE}/3rdparty")

set(HM_INCLUDE_DIRS
  ${HOUMO_ENGINE_DIR}/include
  ${HM_TOKENIZER_INC}
  ${HM_THIRDPARTY_INC}
  ${TCIM_RUNTIME_INCLUDE}
)

set(HM_LINK_DIRS
  ${HM_LIB_DIR}
  ${TCIM_RUNTIME_PATH}/lib
)

if(MSVC)
  option(HM_ENGINE_AUTO_BUILD "Auto-build houmo_infer when sources change" ON)

  if(HM_ENGINE_AUTO_BUILD AND NOT TARGET houmo_infer_build)
    file(GLOB_RECURSE _hm_engine_src
      ${HOUMO_ENGINE_DIR}/src/*.cc
      ${HOUMO_ENGINE_DIR}/src/*.h
    )
    file(GLOB_RECURSE _hm_engine_inc
      ${HOUMO_ENGINE_DIR}/include/*.h
    )

    add_custom_command(
      OUTPUT
        ${HM_LIB_DIR}/houmo_infer.lib
        ${HM_LIB_DIR}/houmo_infer.dll
        ${HM_LIB_DIR}/tokenizer_lib.lib
        ${HM_LIB_DIR}/tokenizer_lib.dll
      COMMAND cmd /c call "${HOUMO_ENGINE_DIR}/build_win.bat"
      WORKING_DIRECTORY ${HOUMO_ENGINE_DIR}
      DEPENDS
        ${HOUMO_ENGINE_DIR}/build_win.bat
        ${HOUMO_ENGINE_DIR}/CMakeLists.txt
        "${HOUMO_EXAMPLES_PATH_CMAKE}/cmake/tcim_runtime.cmake"
        ${_hm_engine_src}
        ${_hm_engine_inc}
      COMMENT "Building houmo_engine Windows artifacts..."
    )

    add_custom_target(houmo_infer_build
      DEPENDS
        ${HM_LIB_DIR}/houmo_infer.lib
        ${HM_LIB_DIR}/houmo_infer.dll
        ${HM_LIB_DIR}/tokenizer_lib.lib
        ${HM_LIB_DIR}/tokenizer_lib.dll
    )
  endif()

  if(NOT TARGET houmo_infer)
    add_library(houmo_infer SHARED IMPORTED GLOBAL)
    set_target_properties(houmo_infer PROPERTIES
      IMPORTED_IMPLIB "${HM_LIB_DIR}/houmo_infer.lib"
      IMPORTED_LOCATION "${HM_LIB_DIR}/houmo_infer.dll"
    )
  endif()

  if(NOT TARGET tokenizer_lib)
    add_library(tokenizer_lib SHARED IMPORTED GLOBAL)
    set_target_properties(tokenizer_lib PROPERTIES
      IMPORTED_IMPLIB "${HM_LIB_DIR}/tokenizer_lib.lib"
      IMPORTED_LOCATION "${HM_LIB_DIR}/tokenizer_lib.dll"
    )
  endif()

  set(HM_LINK_LIBS houmo_infer tokenizer_lib ${TCIM_LIBRARY})
else()
  set(HM_LINK_LIBS
    houmo_infer
    ${TCIM_LIBRARY}
  )

  if(ANDROID_ABI)
    set(_HM_BUILD_SCRIPT "${HOUMO_ENGINE_DIR}/build_ndk.sh")
  else()
    set(_HM_BUILD_SCRIPT "${HOUMO_ENGINE_DIR}/build_linux.sh")
  endif()
  option(HM_ENGINE_AUTO_BUILD "Auto-build houmo_infer when sources change" ON)

  if(HM_ENGINE_AUTO_BUILD AND NOT TARGET houmo_infer_build)
    file(GLOB_RECURSE _hm_engine_src
      ${HOUMO_ENGINE_DIR}/src/*.cc
      ${HOUMO_ENGINE_DIR}/src/*.h
    )
    file(GLOB_RECURSE _hm_engine_inc
      ${HOUMO_ENGINE_DIR}/include/*.h
    )

    add_custom_command(
      OUTPUT ${HM_LIB_DIR}/libhoumo_infer.so
      COMMAND bash ${_HM_BUILD_SCRIPT}
      WORKING_DIRECTORY ${HOUMO_ENGINE_DIR}
      DEPENDS
        ${_HM_BUILD_SCRIPT}
        ${HOUMO_ENGINE_DIR}/CMakeLists.txt
        "$ENV{HOUMO_EXAMPLES_PATH}/cmake/tcim_runtime.cmake"
        ${_hm_engine_src}
        ${_hm_engine_inc}
      COMMENT "Building houmo_engine (${_HM_BUILD_SCRIPT})..."
    )

    add_custom_target(houmo_infer_build
      DEPENDS ${HM_LIB_DIR}/libhoumo_infer.so
    )
  endif()
endif()

add_library(houmo_engine INTERFACE)
target_include_directories(houmo_engine INTERFACE ${HM_INCLUDE_DIRS})
target_link_directories(houmo_engine INTERFACE ${HM_LINK_DIRS})
target_link_libraries(houmo_engine INTERFACE ${HM_LINK_LIBS})

if(HM_ENGINE_AUTO_BUILD AND TARGET houmo_infer_build)
  add_dependencies(houmo_engine houmo_infer_build)
endif()
