# Audio libraries prebuilt configuration
# Use prebuilt audio libraries from 3rdparty/audio_3rdparty

set(AUDIO_THIRDPARTY_PATH ${CMAKE_CURRENT_SOURCE_DIR}/3rdparty/audio_3rdparty)

# Include directories for audio libraries
set(AUDIO_INCLUDE_DIRS
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/include
)

# Library directories for audio libraries
link_directories(${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib)

# Audio libraries to link
set(AUDIO_LIBS samplerate sndfile kaldi-native-fbank-core)

# Audio libraries to install (prebuilt files)
set(AUDIO_LIB_FILES
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib/libsamplerate.so
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib/libsndfile.so
    ${AUDIO_THIRDPARTY_PATH}/${LIB_PATH}/lib/libkaldi-native-fbank-core.so
)
set(AUDIO_INSTALL_COMMAND "install(FILES \${AUDIO_LIB_FILES} DESTINATION .)")