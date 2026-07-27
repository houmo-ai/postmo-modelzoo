# Audio libraries source build configuration
# Build audio libraries (libsamplerate, libsndfile, kaldi-native-fbank) from source

set(AUDIO_THIRDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/audio_source)

# Set install directories to root (not lib/ and include/)
set(CMAKE_INSTALL_LIBDIR "." CACHE STRING "Library install directory" FORCE)
set(CMAKE_INSTALL_INCLUDEDIR "include" CACHE STRING "Include install directory" FORCE)
set(CMAKE_INSTALL_BINDIR "bin" CACHE STRING "Binary install directory" FORCE)

# Disable extra installs for third-party libraries
set(KALDI_NATIVE_FBANK_BUILD_TESTS OFF CACHE BOOL "Disable kaldi tests" FORCE)
set(KALDI_NATIVE_FBANK_BUILD_PYTHON OFF CACHE BOOL "Disable kaldi python" FORCE)
set(BUILD_TESTING OFF CACHE BOOL "Disable testing" FORCE)

# libsndfile options - disable programs, docs, cmake config, pkgconfig, manpages
set(ENABLE_PROGRAMS OFF CACHE BOOL "Disable sndfile programs" FORCE)
set(ENABLE_CPACK OFF CACHE BOOL "Disable CPack" FORCE)
set(ENABLE_PACKAGE_CONFIG OFF CACHE BOOL "Disable package config" FORCE)
set(INSTALL_PKGCONFIG_MODULE OFF CACHE BOOL "Disable pkgconfig install" FORCE)
set(INSTALL_MANPAGES OFF CACHE BOOL "Disable man pages" FORCE)

# libsamplerate options
set(LIBSAMPLERATE_ENABLE_TESTS OFF CACHE BOOL "Disable samplerate tests" FORCE)

# Use EXCLUDE_FROM_ALL to prevent auto-install of third-party libs
add_subdirectory(${AUDIO_THIRDPARTY_PATH}/kaldi-native-fbank ${CMAKE_CURRENT_BINARY_DIR}/kaldi-native-fbank EXCLUDE_FROM_ALL)
add_subdirectory(${AUDIO_THIRDPARTY_PATH}/libsndfile ${CMAKE_CURRENT_BINARY_DIR}/libsndfile EXCLUDE_FROM_ALL)
add_subdirectory(${AUDIO_THIRDPARTY_PATH}/libsamplerate ${CMAKE_CURRENT_BINARY_DIR}/libsamplerate EXCLUDE_FROM_ALL)

# Include directories for audio libraries
set(AUDIO_INCLUDE_DIRS
    ${AUDIO_THIRDPARTY_PATH}/kaldi-native-fbank
    ${AUDIO_THIRDPARTY_PATH}/libsamplerate/include
    ${AUDIO_THIRDPARTY_PATH}/libsndfile/include
)

# Audio libraries to link
set(AUDIO_LIBS samplerate sndfile kaldi-native-fbank-core)

# Audio libraries to install
set(AUDIO_INSTALL_COMMAND "install(TARGETS samplerate sndfile kaldi-native-fbank-core LIBRARY DESTINATION .)")