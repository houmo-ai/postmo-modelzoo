if(TARGET yaml-cpp)
  return()
endif()

if(NOT DEFINED ENV{HOUMO_EXAMPLES_PATH})
  message(FATAL_ERROR "HOUMO_EXAMPLES_PATH not set")
endif()

file(TO_CMAKE_PATH "$ENV{HOUMO_EXAMPLES_PATH}" HOUMO_EXAMPLES_PATH_CMAKE)
set(YAML_CPP_SOURCE_DIR "${HOUMO_EXAMPLES_PATH_CMAKE}/3rdparty/yaml-cpp")
if(NOT EXISTS "${YAML_CPP_SOURCE_DIR}/include/yaml-cpp/yaml.h")
  message(FATAL_ERROR "yaml-cpp sources not found: ${YAML_CPP_SOURCE_DIR}")
endif()

option(YAML_CPP_BUILD_CONTRIB "Enable yaml-cpp contrib in library" ON)

file(GLOB YAML_CPP_SOURCES CONFIGURE_DEPENDS
  "${YAML_CPP_SOURCE_DIR}/src/*.cpp"
)
if(YAML_CPP_BUILD_CONTRIB)
  file(GLOB YAML_CPP_CONTRIB_SOURCES CONFIGURE_DEPENDS
    "${YAML_CPP_SOURCE_DIR}/src/contrib/*.cpp"
  )
  list(APPEND YAML_CPP_SOURCES ${YAML_CPP_CONTRIB_SOURCES})
endif()

add_library(yaml-cpp SHARED ${YAML_CPP_SOURCES})
add_library(yaml-cpp::yaml-cpp ALIAS yaml-cpp)

if(ANDROID_ABI)
  set(YAML_CPP_OUTPUT_DIR "${HOUMO_EXAMPLES_PATH_CMAKE}/utils/android")
else()
  set(YAML_CPP_OUTPUT_DIR "${HOUMO_EXAMPLES_PATH_CMAKE}/utils/lib")
endif()

set_target_properties(yaml-cpp PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY "$<1:${YAML_CPP_OUTPUT_DIR}>"
  LIBRARY_OUTPUT_DIRECTORY "$<1:${YAML_CPP_OUTPUT_DIR}>"
  ARCHIVE_OUTPUT_DIRECTORY "$<1:${YAML_CPP_OUTPUT_DIR}>"
)

target_include_directories(yaml-cpp
  PUBLIC ${YAML_CPP_SOURCE_DIR}/include
  PRIVATE ${YAML_CPP_SOURCE_DIR}/src
)
target_compile_features(yaml-cpp PUBLIC cxx_std_11)

target_compile_definitions(yaml-cpp
  PRIVATE
    YAML_CPP_DLL
    $<$<NOT:$<BOOL:${YAML_CPP_BUILD_CONTRIB}>>:YAML_CPP_NO_CONTRIB>
)

if(MSVC)
  target_compile_options(yaml-cpp PRIVATE /W3 /wd4127 /wd4355)
endif()

if(DEFINED LIB_INSTALL_PATH AND NOT "${LIB_INSTALL_PATH}" STREQUAL "")
  set(YAML_CPP_INSTALL_DIR "${LIB_INSTALL_PATH}")
else()
  set(YAML_CPP_INSTALL_DIR "${YAML_CPP_OUTPUT_DIR}")
endif()

install(TARGETS yaml-cpp
  RUNTIME DESTINATION "${YAML_CPP_INSTALL_DIR}"
  LIBRARY DESTINATION "${YAML_CPP_INSTALL_DIR}"
  ARCHIVE DESTINATION "${YAML_CPP_INSTALL_DIR}"
)
