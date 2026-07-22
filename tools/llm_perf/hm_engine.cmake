# hm_engine.cmake — reusable CMake module for houmo-llm-engine
#
# Usage:
#   include(path/to/hm_engine.cmake)
#   target_link_libraries(your_target PRIVATE hm_engine)
#
# Sets up:
#   - hm_engine INTERFACE library (includes + link dirs + link libs)
#   - houmo_infer_build custom target on Linux/Android when auto-build is enabled

if(_HM_ENGINE_LOADED)
  return()
endif()
set(_HM_ENGINE_LOADED TRUE)

if(NOT DEFINED ENV{HOUMO_EXAMPLES_PATH})
  message(FATAL_ERROR "HOUMO_EXAMPLES_PATH not set")
endif()

if(NOT DEFINED ENV{TCIM_RUNTIME_PATH})
  set(ENV{TCIM_RUNTIME_PATH} "$ENV{HOUMO_EXAMPLES_PATH}/apis/common/tcim_runtime")
endif()

file(TO_CMAKE_PATH "$ENV{HOUMO_EXAMPLES_PATH}" HM_EXAMPLES_ROOT)
file(TO_CMAKE_PATH "$ENV{TCIM_RUNTIME_PATH}" HM_TCIM_RUNTIME_ROOT)

set(HM_LLM_ENGINE_DIR "${HM_EXAMPLES_ROOT}/tools/common/houmo-llm-engine")
if(ANDROID_ABI)
  set(HM_LIB_DIR "${HM_EXAMPLES_ROOT}/tools/common/android")
else()
  set(HM_LIB_DIR "${HM_EXAMPLES_ROOT}/tools/common/lib")
endif()

set(HM_TOKENIZER_INC "${HM_EXAMPLES_ROOT}/apis/common/tokenizer.cpp/include")
set(HM_HALF_INC "${HM_EXAMPLES_ROOT}/apis/common/hpp")
set(HM_TCIM_INC "${HM_TCIM_RUNTIME_ROOT}/include")
set(HM_TCIM_LIB_DIR "${HM_TCIM_RUNTIME_ROOT}/lib")
set(HM_EIGEN_INC "${HM_EXAMPLES_ROOT}/apis/common")

set(HM_INCLUDE_DIRS
  ${HM_LLM_ENGINE_DIR}/include
  ${HM_TOKENIZER_INC}
  ${HM_HALF_INC}
  ${HM_TCIM_INC}
  ${HM_EIGEN_INC}
)

set(HM_LINK_DIRS
  ${HM_LIB_DIR}
  ${HM_TCIM_LIB_DIR}
)

if(MSVC)
  option(HM_ENGINE_AUTO_BUILD "Auto-build houmo_infer when sources change" ON)

  if(HM_ENGINE_AUTO_BUILD AND NOT TARGET houmo_infer_build)
    file(GLOB_RECURSE _hm_engine_src
      ${HM_LLM_ENGINE_DIR}/src/*.cc
      ${HM_LLM_ENGINE_DIR}/src/*.h
    )
    file(GLOB_RECURSE _hm_engine_inc
      ${HM_LLM_ENGINE_DIR}/include/*.h
    )

    add_custom_command(
      OUTPUT
        ${HM_LIB_DIR}/houmo_infer.lib
        ${HM_LIB_DIR}/houmo_infer.dll
        ${HM_LIB_DIR}/tokenizer_lib.lib
        ${HM_LIB_DIR}/tokenizer_lib.dll
      COMMAND cmd /c build_win.bat
      WORKING_DIRECTORY ${HM_LLM_ENGINE_DIR}
      DEPENDS
        ${HM_LLM_ENGINE_DIR}/build_win.bat
        ${HM_LLM_ENGINE_DIR}/CMakeLists.txt
        ${HM_LLM_ENGINE_DIR}/tcim_runtime.cmake
        ${_hm_engine_src}
        ${_hm_engine_inc}
      COMMENT "Building houmo-llm-engine Windows artifacts..."
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

  set(HM_LINK_LIBS houmo_infer tokenizer_lib tcim_runtime_lite)
else()
  set(HM_LINK_LIBS
    houmo_infer
    tcim_runtime_lite
  )

  if(ANDROID_ABI)
    set(_HM_BUILD_SCRIPT "${HM_LLM_ENGINE_DIR}/build_ndk.sh")
  else()
    set(_HM_BUILD_SCRIPT "${HM_LLM_ENGINE_DIR}/build_linux.sh")
  endif()
  option(HM_ENGINE_AUTO_BUILD "Auto-build houmo_infer when sources change" ON)

  if(HM_ENGINE_AUTO_BUILD AND NOT TARGET houmo_infer_build)
    file(GLOB_RECURSE _hm_engine_src
      ${HM_LLM_ENGINE_DIR}/src/*.cc
      ${HM_LLM_ENGINE_DIR}/src/*.h
    )
    file(GLOB_RECURSE _hm_engine_inc
      ${HM_LLM_ENGINE_DIR}/include/*.h
    )

    add_custom_command(
      OUTPUT ${HM_LIB_DIR}/libhoumo_infer.so
      COMMAND bash ${_HM_BUILD_SCRIPT}
      WORKING_DIRECTORY ${HM_LLM_ENGINE_DIR}
      DEPENDS
        ${_HM_BUILD_SCRIPT}
        ${HM_LLM_ENGINE_DIR}/CMakeLists.txt
        ${HM_LLM_ENGINE_DIR}/tcim_runtime.cmake
        ${_hm_engine_src}
        ${_hm_engine_inc}
      COMMENT "Building houmo-llm-engine (${_HM_BUILD_SCRIPT})..."
    )

    add_custom_target(houmo_infer_build
      DEPENDS ${HM_LIB_DIR}/libhoumo_infer.so
    )
  endif()
endif()

add_library(hm_engine INTERFACE)
target_include_directories(hm_engine INTERFACE ${HM_INCLUDE_DIRS})
target_link_directories(hm_engine INTERFACE ${HM_LINK_DIRS})
target_link_libraries(hm_engine INTERFACE ${HM_LINK_LIBS})

if(HM_ENGINE_AUTO_BUILD AND TARGET houmo_infer_build)
  add_dependencies(hm_engine houmo_infer_build)
endif()
