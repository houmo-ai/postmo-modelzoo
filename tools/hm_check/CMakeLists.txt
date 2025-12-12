cmake_minimum_required(VERSION 3.8)
project(hm-check C CXX)

include($ENV{HOUMO_EXAMPLES_PATH}/tools/tcim_runtime.cmake)
include_directories($ENV{HOUMO_EXAMPLES_PATH}/apis/common/nlohmann/include)
include_directories($ENV{HOUMO_EXAMPLES_PATH}/apis/common/)
include_directories($ENV{HOUMO_SDK_PATH}/hal/export/include)
link_directories($ENV{HOUMO_SDK_PATH}/hal/lib)

file(GLOB MODEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/models/*.hmm")
set(CMAKE_POSITION_INDEPENDENT_CODE OFF)
set(MODEL_OBJECTS)
foreach(M ${MODEL_FILES})
    get_filename_component(M_NAME ${M} NAME_WE)   # model1
    get_filename_component(M_DIR ${M} DIRECTORY)
    set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/${M_NAME}.o)
    message(STATUS "Embedding model: ${M_NAME}.hmm")
    add_custom_command(
        OUTPUT ${OBJ_FILE}
        COMMAND ${CMAKE_LINKER} -r -b binary ${M_NAME}.hmm -o ${OBJ_FILE}
        COMMAND ${CMAKE_OBJCOPY} --add-section .note.GNU-stack=/dev/null
                             --set-section-flags .note.GNU-stack=readonly ${OBJ_FILE}
        DEPENDS ${M}
        WORKING_DIRECTORY ${M_DIR}
        COMMENT "Embedding ${M_NAME}.hmm into ${OBJ_FILE}"
    )
    list(APPEND MODEL_OBJECTS ${OBJ_FILE})
endforeach()

add_executable(${CMAKE_PROJECT_NAME} main.cc models.cc ${MODEL_OBJECTS})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ${TCIM_LIBRARY} dl)
if(ANDROID)
else()
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE pthread hal_xh2a)
endif()
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)