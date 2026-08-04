get_filename_component(
  QWEN3_TTS_FRONTEND_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)

function(add_qwen3_tts_streaming_frontend target_name)
  cmake_parse_arguments(ARG "WITH_TEXT_PROCESSOR" "" "" ${ARGN})

  set(frontend_sources
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_code_predictor.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_code_predictor_embedding.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_codec_embedding.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_prefill_decode_runtime.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_sampler.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_streaming_prompt_builder.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_streaming_generator.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_talker.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_stateful_decoder.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_wav_writer.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_text_embedding.cc"
      "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_text_projection.cc")

  if(ARG_WITH_TEXT_PROCESSOR)
    list(APPEND frontend_sources
         "${QWEN3_TTS_FRONTEND_ROOT}/src/qwen3_tts_text_processor.cc")
  endif()

  add_library(${target_name} STATIC ${frontend_sources})
  target_include_directories(
    ${target_name} PUBLIC "${QWEN3_TTS_FRONTEND_ROOT}/include")
  target_link_libraries(${target_name} PUBLIC houmo_engine)

  if(ANDROID_ABI)
    target_link_libraries(${target_name} PUBLIC dl m)
  endif()
endfunction()
