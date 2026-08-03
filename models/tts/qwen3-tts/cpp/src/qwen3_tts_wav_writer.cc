/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_wav_writer.cc
 * Description:
 *   Qwen3-TTS PCM WAV output writer implementation.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include "qwen3_tts_wav_writer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace houmo {
namespace {

template <typename T>
void WriteValue(std::fstream* stream, T value) {
  stream->write(reinterpret_cast<const char*>(&value), sizeof(value));
}

}  // namespace

Qwen3TTSWavWriter::Qwen3TTSWavWriter(const std::string& path,
                                     uint32_t sample_rate)
    : stream_(path, std::ios::binary | std::ios::in | std::ios::out |
                       std::ios::trunc),
      sample_rate_(sample_rate) {
  if (!stream_) throw std::runtime_error("Failed to open WAV output: " + path);
  WriteHeader(0);
}

Qwen3TTSWavWriter::~Qwen3TTSWavWriter() {
  try {
    Close();
  } catch (...) {
  }
}

void Qwen3TTSWavWriter::WriteHeader(uint32_t data_bytes) {
  stream_.seekp(0);
  stream_.write("RIFF", 4);
  WriteValue(&stream_, static_cast<uint32_t>(36 + data_bytes));
  stream_.write("WAVEfmt ", 8);
  WriteValue(&stream_, static_cast<uint32_t>(16));
  WriteValue(&stream_, static_cast<uint16_t>(1));
  WriteValue(&stream_, static_cast<uint16_t>(1));
  WriteValue(&stream_, sample_rate_);
  WriteValue(&stream_, static_cast<uint32_t>(sample_rate_ * 2));
  WriteValue(&stream_, static_cast<uint16_t>(2));
  WriteValue(&stream_, static_cast<uint16_t>(16));
  stream_.write("data", 4);
  WriteValue(&stream_, data_bytes);
}

void Qwen3TTSWavWriter::Write(const std::vector<float>& samples) {
  if (closed_) throw std::runtime_error("Cannot write a closed WAV file");
  stream_.seekp(0, std::ios::end);
  for (float sample : samples) {
    const float clipped = std::clamp(sample, -1.0f, 1.0f);
    const auto pcm = static_cast<int16_t>(
        std::lrint(clipped * std::numeric_limits<int16_t>::max()));
    WriteValue(&stream_, pcm);
  }
  sample_count_ += samples.size();
}

void Qwen3TTSWavWriter::Close() {
  if (closed_) return;
  const uint64_t bytes = sample_count_ * sizeof(int16_t);
  if (bytes > std::numeric_limits<uint32_t>::max()) {
    throw std::runtime_error("WAV output exceeds RIFF size limit");
  }
  WriteHeader(static_cast<uint32_t>(bytes));
  stream_.flush();
  stream_.close();
  closed_ = true;
}

}  // namespace houmo
