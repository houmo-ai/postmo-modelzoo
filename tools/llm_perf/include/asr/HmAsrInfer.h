/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: HmAsrInfer.h
 * Description:
 *   ASR inference wrapper using libhoumo_infer.a
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef HM_ASR_INFER_H
#define HM_ASR_INFER_H

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "base/houmo.h"
#include "core/asr_model.h"
#include "core/model_factory.h"

enum class AsrModelType {
  Whisper,
  GlmAsr,
  Qwen3Asr
};

struct AsrTranscribeResult {
  double audio_load_time_ms = 0.0;
  double encode_time_ms = 0.0;
  double prefill_time_ms = 0.0;
  double decode_time_ms = 0.0;
  double total_time_ms = 0.0;
  double ttft_ms = 0.0;
  float audio_duration_s = 0.0f;
  float overall_rtf = 0.0f;
  float inference_rtf = 0.0f;
  float decode_tps = 0.0f;
  float overall_tps = 0.0f;
  int output_tokens = 0;
};

class HmAsrInfer {
 public:
  HmAsrInfer(AsrModelType type,
             const std::string& encode_path,
             const std::string& prefill_path,
             const std::string& decode_path,
             const std::string& tokenizer_path,
             const std::string& embedding_path,
             const std::vector<int>& devices);

  ~HmAsrInfer();

  HmAsrInfer(const HmAsrInfer&) = delete;
  HmAsrInfer& operator=(const HmAsrInfer&) = delete;

  AsrTranscribeResult Transcribe(const std::string& audio_path);

  double GetLoadTimeMs() const { return load_time_ms_; }

  static AsrModelType DetectModelType(const std::string& prefill_path,
                                       const std::vector<int>& devices);
  static const char* ModelTypeToString(AsrModelType type);

 private:
  AsrModelType type_;
  std::unique_ptr<houmo::ASRModel> model_;
  std::unique_ptr<houmo::Context> ctx_;
  double load_time_ms_ = 0.0;
};

#endif  // HM_ASR_INFER_H
