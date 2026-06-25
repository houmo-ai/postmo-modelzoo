# hm_engine.cmake — reusable CMake module for houmo-llm-engine
#
# Usage:
#   include(path/to/hm_engine.cmake)
#   target_link_libraries(your_target PRIVATE hm_engine)
#
# Sets up:
#   - hm_engine  INTERFACE library (includes + link dirs + link libs)
#   - houmo_infer_build  custom target (auto-builds libhoumo_infer.so)
#
# Options (set before include):
#   HM_ENGINE_AUTO_BUILD  ON|OFF  (default ON)

if(_HM_ENGINE_LOADED)
  return()
endif()
set(_HM_ENGINE_LOADED TRUE)

# ---- validate env ----
if(NOT DEFINED ENV{HOUMO_EXAMPLES_PATH})
  message(FATAL_ERROR "HOUMO_EXAMPLES_PATH not set")
endif()

if(NOT DEFINED ENV{TCIM_RUNTIME_PATH})
  set(ENV{TCIM_RUNTIME_PATH} "$ENV{HOUMO_EXAMPLES_PATH}/apis/common/tcim_runtime")
endif()

# ---- paths ----
set(HM_LLM_ENGINE_DIR  "$ENV{HOUMO_EXAMPLES_PATH}/tools/common/houmo-llm-engine")
if(ANDROID_ABI)
  set(HM_LIB_DIR        "$ENV{HOUMO_EXAMPLES_PATH}/tools/common/android")
else()
  set(HM_LIB_DIR        "$ENV{HOUMO_EXAMPLES_PATH}/tools/common/lib")
endif()
set(HM_TOKENIZER_INC    "$ENV{HOUMO_EXAMPLES_PATH}/apis/common/tokenizer.cpp/include")
set(HM_HALF_INC         "$ENV{HOUMO_EXAMPLES_PATH}/apis/common/half/include")
set(HM_TCIM_INC         "$ENV{TCIM_RUNTIME_PATH}/include")
set(HM_TCIM_LIB_DIR     "$ENV{TCIM_RUNTIME_PATH}/lib")
set(HM_EIGEN_INC        "${HM_LLM_ENGINE_DIR}/3rdparty")

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

set(HM_LINK_LIBS
  houmo_infer
  tcim_runtime_lite
)

# ---- auto-build houmo_infer (optional) ----
if(ANDROID_ABI)
  set(_HM_BUILD_SCRIPT  "${HM_LLM_ENGINE_DIR}/build_ndk.sh")
  option(HM_ENGINE_AUTO_BUILD "Auto-build houmo_infer when sources change" OFF)
else()
  set(_HM_BUILD_SCRIPT  "${HM_LLM_ENGINE_DIR}/build_linux.sh")
  option(HM_ENGINE_AUTO_BUILD "Auto-build houmo_infer when sources change" ON)
endif()

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

# ---- INTERFACE library ----
add_library(hm_engine INTERFACE)
target_include_directories(hm_engine INTERFACE ${HM_INCLUDE_DIRS})
target_link_directories(hm_engine INTERFACE ${HM_LINK_DIRS})
target_link_libraries(hm_engine INTERFACE ${HM_LINK_LIBS})

if(HM_ENGINE_AUTO_BUILD)
  add_dependencies(hm_engine houmo_infer_build)
endif()
