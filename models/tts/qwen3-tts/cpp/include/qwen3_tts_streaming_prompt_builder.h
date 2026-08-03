/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: qwen3_tts_streaming_prompt_builder.h
 * Description:
 *   Qwen3-TTS streaming Talker prompt construction interface.
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

#pragma once

#include "qwen3_tts_types.h"

namespace houmo {

struct Qwen3TTSStreamingPrompt {
  Qwen3TTSHiddenSequence initial_prompt;
  Qwen3TTSHiddenSequence trailing_text_hidden;
  Qwen3TTSHiddenSequence text_pad_hidden;
};

class Qwen3TTSStreamingPromptBuilder {
 public:
  Qwen3TTSStreamingPrompt Build(
      const Qwen3TTSHiddenSequence& role_hidden,
      const Qwen3TTSHiddenSequence& body_hidden,
      const Qwen3TTSHiddenSequence& tts_bos_hidden,
      const Qwen3TTSHiddenSequence& tts_eos_hidden,
      const Qwen3TTSHiddenSequence& tts_pad_hidden,
      const Qwen3TTSHiddenSequence& codec_prompt_hidden) const;
};

}  // namespace houmo
