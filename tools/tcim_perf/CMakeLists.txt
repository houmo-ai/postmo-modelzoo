cmake_minimum_required(VERSION 3.8)
project(tcim_perf)

include($ENV{HOUMO_EXAMPLES_PATH}/tools/tcim_runtime.cmake)
include_directories(
  $ENV{HOUMO_SDK_PATH}/hal/export/include
  $ENV{HOUMO_SDK_PATH}/hal/include
  $ENV{HOUMO_EXAMPLES_PATH}/apis/common/nlohmann/include
)
link_directories(
  $ENV{HOUMO_SDK_PATH}/hal/lib
)
if(MSVC)
    add_compile_options("/std:c++17" "/utf-8")
endif()
add_executable(${CMAKE_PROJECT_NAME} tcim_perf.cc)
if(MSVC)
  target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} ws2_32 Userenv ntdll Bcrypt kernel32)
else()
  target_link_libraries(${CMAKE_PROJECT_NAME} ${TCIM_LIBRARY} dl)
endif()

if(MSVC)
else()
    if(ANDROID)
    else()
        target_link_libraries(${CMAKE_PROJECT_NAME} pthread hal_xh2a)
        target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE XH2A_HM_SYS)
    endif()
endif()

install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)