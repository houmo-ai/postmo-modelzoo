# Android platform configuration for houmo_infer.

set(HOUMO_PLATFORM_NAME "android")
set(HOUMO_NEED_PTHREAD FALSE)
set(HOUMO_CXX17_FLAG "-std=c++17")
set(HOUMO_COMPILE_OPTIONS -Wno-deprecated-declarations)
set(HOUMO_RELEASE_FLAGS "-O3 -DNDEBUG -DEIGEN_MPL2_ONLY")
set(HOUMO_DEBUG_FLAGS "-g -O0 -ggdb")
set(HOUMO_EXPORT_ALL_SYMBOLS FALSE)
set(HOUMO_TARGET_LINK_OPTIONS "")
set(LIB_PATH Android_xh2)

if(DEFINED ENV{HOUMO_EXAMPLES_PATH})
  set(LIB_INSTALL_PATH "$ENV{HOUMO_EXAMPLES_PATH}/utils/android")
endif()
