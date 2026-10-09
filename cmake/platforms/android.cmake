# Android platform configuration for houmo_infer.

set(POSTMO_PLATFORM_NAME "android")
set(POSTMO_NEED_PTHREAD FALSE)
set(POSTMO_CXX17_FLAG "-std=c++17")
set(POSTMO_COMPILE_OPTIONS -Wno-deprecated-declarations)
set(POSTMO_RELEASE_FLAGS "-O3 -DNDEBUG -DEIGEN_MPL2_ONLY")
set(POSTMO_DEBUG_FLAGS "-g -O0 -ggdb")
set(POSTMO_EXPORT_ALL_SYMBOLS FALSE)
set(POSTMO_TARGET_LINK_OPTIONS "")
set(LIB_PATH Android_xh2)

if(DEFINED ENV{POSTMO_EXAMPLES_PATH})
  set(LIB_INSTALL_PATH "$ENV{POSTMO_EXAMPLES_PATH}/utils/android")
endif()
