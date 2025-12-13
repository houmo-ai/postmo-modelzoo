cmake_minimum_required(VERSION 3.8)
project(hm-check C CXX)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /std:c++17 /W1 /utf-8 /O2 /DNDEBUG")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -w -O2")
endif()

include_directories($ENV{TCIM_RUNTIME_PATH}/include)
include_directories($ENV{HOUMO_SDK_PATH}/hal/export/include)

link_directories($ENV{TCIM_RUNTIME_PATH}/lib)
link_directories($ENV{HOUMO_SDK_PATH}/hal/lib)

set(MODEL_OBJECTS)
if(MSVC)
    enable_language(RC)
    list(APPEND MODEL_OBJECTS resource.rc)
else()
    file(GLOB MODEL_FILES "${CMAKE_CURRENT_SOURCE_DIR}/models/*.hmm")
    foreach(M ${MODEL_FILES})
        get_filename_component(M_NAME ${M} NAME_WE)   # model1
        get_filename_component(M_DIR ${M} DIRECTORY)
        set(OBJ_FILE ${CMAKE_CURRENT_BINARY_DIR}/${M_NAME}.o)
        message(STATUS "Embedding model: ${M_NAME}.hmm")
        # For cross-compiling (e.g. Android) ld may not accept -b binary without
        # an emulation target. Prefer using objcopy to create an object
        # with the correct ELF target when possible.
        if(CMAKE_OBJCOPY)
            if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
                set(_OBJ_OUTPUT_FMT "elf64-littleaarch64")
            elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|amd64")
                set(_OBJ_OUTPUT_FMT "elf64-x86-64")
            else()
                set(_OBJ_OUTPUT_FMT "")
            endif()
        else()
            set(_OBJ_OUTPUT_FMT "")
        endif()

        if(_OBJ_OUTPUT_FMT)
            add_custom_command(
                OUTPUT ${OBJ_FILE}
                COMMAND ${CMAKE_OBJCOPY} --input-target=binary --output-target=${_OBJ_OUTPUT_FMT} ${M_NAME}.hmm ${OBJ_FILE}
                COMMAND ${CMAKE_OBJCOPY} --add-section .note.GNU-stack=/dev/null --set-section-flags .note.GNU-stack=readonly ${OBJ_FILE}
                DEPENDS ${M}
                WORKING_DIRECTORY ${M_DIR}
                COMMENT "Embedding ${M_NAME}.hmm into ${OBJ_FILE} (via objcopy)"
            )
        else()
            add_custom_command(
                OUTPUT ${OBJ_FILE}
                COMMAND ${CMAKE_LINKER} -r -b binary ${M_NAME}.hmm -o ${OBJ_FILE}
                COMMAND ${CMAKE_OBJCOPY} --add-section .note.GNU-stack=/dev/null
                                     --set-section-flags .note.GNU-stack=readonly ${OBJ_FILE}
                DEPENDS ${M}
                WORKING_DIRECTORY ${M_DIR}
                COMMENT "Embedding ${M_NAME}.hmm into ${OBJ_FILE}"
            )
        endif()
        list(APPEND MODEL_OBJECTS ${OBJ_FILE})
    endforeach()
endif()


add_executable(${CMAKE_PROJECT_NAME} main.cc models.cc ${MODEL_OBJECTS})
target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE tcim_runtime_lite hal_xh2a)
if(MSVC)
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE ws2_32 kernel32)
else()
    if(NOT ANDROID)
        target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE pthread)
    endif()
    target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE dl)
endif()
install(TARGETS ${CMAKE_PROJECT_NAME} DESTINATION .)