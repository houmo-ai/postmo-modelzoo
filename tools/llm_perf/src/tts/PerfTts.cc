/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: PerfTts.cc
 * Description:
 *   Qwen3-TTS fixed-frame performance inference implementation.
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

#include "tts/PerfTts.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <limits>
#include <memory>
#include <random>
#include <sstream>
#include <stdexcept>
#include <utility>

#include "qwen3_tts_code_predictor.h"
#include "qwen3_tts_code_predictor_embedding.h"
#include "qwen3_tts_codec_embedding.h"
#include "qwen3_tts_sampler.h"
#include "qwen3_tts_stateful_decoder.h"
#include "qwen3_tts_streaming_generator.h"
#include "qwen3_tts_streaming_prompt_builder.h"
#include "qwen3_tts_talker.h"
#include "qwen3_tts_text_embedding.h"
#include "qwen3_tts_text_projection.h"

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::array<houmo::Token, 3> kRoleTokens = {151644, 77091, 198};
constexpr houmo::Token kTtsPadToken = 151671;
constexpr houmo::Token kTtsBosToken = 151672;
constexpr houmo::Token kTtsEosToken = 151673;
constexpr houmo::Token kCodecPadToken = 2148;
constexpr houmo::Token kCodecBosToken = 2149;
constexpr houmo::Token kCodecThinkToken = 2154;
constexpr houmo::Token kCodecThinkBosToken = 2156;
constexpr houmo::Token kCodecThinkEosToken = 2157;
constexpr houmo::Token kChineseToken = 2055;
constexpr houmo::Token kVivianToken = 3065;
constexpr houmo::Token kCodecEosToken = 2150;
constexpr size_t kInitialPromptLength = 10;

const std::array<houmo::Token, 5> kExcludedTextTokens = {
    151644, 151645, kTtsPadToken, kTtsBosToken, kTtsEosToken};

double Milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}

size_t CheckedCeil(long double value, const char* name) {
  if (!std::isfinite(value) || value < 0.0L ||
      value > static_cast<long double>(std::numeric_limits<size_t>::max())) {
    throw std::invalid_argument(std::string(name) + " is out of range");
  }
  return static_cast<size_t>(std::ceil(value));
}

size_t StableTargetFrames(double requested_audio_length_s) {
  const long double frames =
      static_cast<long double>(requested_audio_length_s) /
      static_cast<long double>(PerfTts::kSecondsPerFrame);
  const long double nearest = std::round(frames);
  const long double tolerance = 16.0L * std::numeric_limits<double>::epsilon() *
                                std::max(1.0L, std::fabs(frames));
  return CheckedCeil(
      std::fabs(frames - nearest) <= tolerance ? nearest : frames,
      "target codec frames");
}

size_t CheckedMultiply(size_t left, size_t right, const char* name) {
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) {
    throw std::invalid_argument(std::string(name) + " is out of range");
  }
  return left * right;
}

size_t CheckedAdd(size_t left, size_t right, const char* name) {
  if (left > std::numeric_limits<size_t>::max() - right) {
    throw std::invalid_argument(std::string(name) + " is out of range");
  }
  return left + right;
}

void RequireFile(const char* name, const std::string& path) {
  if (path.empty() || !std::filesystem::is_regular_file(path)) {
    throw std::invalid_argument(std::string(name) + " does not exist: " + path);
  }
}

houmo::Qwen3TTSHiddenSequence TokenAt(
    const houmo::Qwen3TTSHiddenSequence& sequence, size_t index) {
  sequence.Validate();
  if (index >= sequence.sequence_length) {
    throw std::out_of_range("Projected special token index is out of range");
  }
  houmo::Qwen3TTSHiddenSequence token;
  token.sequence_length = 1;
  token.hidden_dim = sequence.hidden_dim;
  const auto begin = sequence.data.begin() +
                     static_cast<std::ptrdiff_t>(index * sequence.hidden_dim);
  token.data.assign(begin,
                    begin + static_cast<std::ptrdiff_t>(sequence.hidden_dim));
  return token;
}

houmo::Qwen3TTSSamplingConfig TalkerSamplingConfig() {
  houmo::Qwen3TTSSamplingConfig config;
  config.do_sample = true;
  config.temperature = 0.9f;
  config.top_k = 50;
  config.repetition_penalty = 1.05f;
  config.eos_token_id = kCodecEosToken;
  config.seed = PerfTts::kSeed;
  config.suppress_tokens.reserve(1024);
  for (houmo::Token token = 2048; token <= 3071; ++token) {
    config.suppress_tokens.push_back(token);
  }
  return config;
}

houmo::Qwen3TTSSamplingConfig PredictorSamplingConfig() {
  houmo::Qwen3TTSSamplingConfig config;
  config.do_sample = true;
  config.temperature = 0.9f;
  config.top_k = 50;
  config.seed = PerfTts::kSeed;
  return config;
}

}  // namespace

class PerfTts::Impl {
 public:
  explicit Impl(const TtsPerfSettings& settings) : settings_(settings) {
    NormalizeSettings();
    ValidatePathsAndFixedSettings();

    const size_t max_text_sequence =
        std::max<size_t>(settings_.body_text_tokens, kRoleTokens.size() + 3);
    if (max_text_sequence >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument(
          "Text embedding sequence length is out of range");
    }

    text_embedding_ = std::make_unique<houmo::Qwen3TTSTextEmbedding>(
        settings_.text_embedding_path, static_cast<int>(max_text_sequence));
    text_projection_ = std::make_unique<houmo::Qwen3TTSTextProjection>(
        settings_.text_projection_path, settings_.device_id);
    talker_ = std::make_unique<houmo::Qwen3TTSTalker>(
        settings_.talker_prefill_path, settings_.talker_decode_path,
        settings_.device_id);
    codec_embedding_ = std::make_unique<houmo::Qwen3TTSCodecEmbedding>(
        settings_.embedding_path, static_cast<int>(talker_->hidden_dim()), 7);
    code_predictor_ = std::make_unique<houmo::Qwen3TTSCodePredictor>(
        settings_.code_predictor_prefill_path,
        settings_.code_predictor_decode_path, settings_.device_id);
    code_predictor_embedding_ =
        std::make_unique<houmo::Qwen3TTSCodePredictorEmbedding>(
            settings_.code_embedding_path, code_predictor_->hidden_dim());
    stateful_decoder_ = std::make_unique<houmo::Qwen3TTSStatefulDecoder>(
        settings_.stateful_decoder_path, settings_.device_id);

    ValidateModelInterfaces();
    body_tokens_ = GenerateBodyTokens();
    ValidateWorkload();
  }

  void ValidateWorkload() const {
    if (settings_.target_codec_frames < 2) {
      throw std::invalid_argument(
          "TTS fixed-frame streaming requires at least 2 codec frames");
    }
    if (settings_.body_text_tokens > settings_.target_codec_frames - 1) {
      throw std::invalid_argument(
          "body_text_tokens must be no greater than target_codec_frames - 1");
    }

    const size_t context_length = talker_->context_length();
    const size_t prompt_length = kInitialPromptLength;
    const size_t max_frames = context_length > prompt_length + 1
                                  ? context_length - prompt_length - 1
                                  : 0;
    if (settings_.target_codec_frames > max_frames) {
      std::ostringstream message;
      message << "Talker context is too short: requested_audio_length="
              << settings_.requested_audio_length_s
              << "s, target_codec_frames=" << settings_.target_codec_frames
              << ", max_supported_codec_frames=" << max_frames
              << ", max_supported_audio_length="
              << max_frames * PerfTts::kSecondsPerFrame << "s";
      throw std::invalid_argument(message.str());
    }
  }

  TtsPerfResult Run(bool keep_waveform,
                    const TtsProgressCallback& progress_callback) {
    TtsPerfResult result;
    const auto run_start = Clock::now();
    auto decoder_state = stateful_decoder_->CreateState();
    auto sampling_rng = std::make_shared<std::mt19937>(PerfTts::kSeed);
    houmo::Qwen3TTSStreamingGenerator generator(
        talker_.get(), codec_embedding_.get(), code_predictor_.get(),
        code_predictor_embedding_.get(),
        houmo::Qwen3TTSSampler(TalkerSamplingConfig(), sampling_rng),
        houmo::Qwen3TTSSampler(PredictorSamplingConfig(), sampling_rng),
        kCodecEosToken);

    const auto embedding_start = Clock::now();
    const auto role_embedding = text_embedding_->Lookup(
        std::vector<houmo::Token>(kRoleTokens.begin(), kRoleTokens.end()));
    const auto body_embedding = text_embedding_->Lookup(body_tokens_);
    const auto special_embedding =
        text_embedding_->Lookup({kTtsBosToken, kTtsEosToken, kTtsPadToken});
    result.stages.text_embedding_ms =
        Milliseconds(Clock::now() - embedding_start);
    result.stages.text_embedding_count = settings_.text_projection_tokens;

    const auto projection_start = Clock::now();
    const auto role_hidden = text_projection_->Project(role_embedding);
    const auto body_hidden = text_projection_->Project(body_embedding);
    const auto special_hidden = text_projection_->Project(special_embedding);
    result.stages.text_projection_ms =
        Milliseconds(Clock::now() - projection_start);
    result.stages.text_projection_count =
        settings_.text_projection_tokens / text_projection_->chunk_length();

    const auto prompt_start = Clock::now();
    const auto codec_prompt_hidden = codec_embedding_->Lookup(
        {kCodecThinkToken, kCodecThinkBosToken, kChineseToken,
         kCodecThinkEosToken, kVivianToken, kCodecPadToken, kCodecBosToken});
    const auto prompt = prompt_builder_.Build(
        role_hidden, body_hidden, TokenAt(special_hidden, 0),
        TokenAt(special_hidden, 1), TokenAt(special_hidden, 2),
        codec_prompt_hidden);
    result.stages.prompt_prepare_ms = Milliseconds(Clock::now() - prompt_start);
    result.stages.prompt_prepare_count = 1;
    if (prompt.initial_prompt.sequence_length != kInitialPromptLength) {
      throw std::runtime_error("Unexpected TTS initial prompt length");
    }

    std::vector<houmo::Qwen3TTSCodecFrame> decoder_frames;
    decoder_frames.reserve(PerfTts::kDecoderChunkSize);
    std::vector<float> waveform;
    waveform.reserve(settings_.expected_audio_samples);
    size_t frames_seen = 0;
    bool first_audio_seen = false;

    const auto emit_audio = [&](std::vector<float> audio) {
      if (audio.empty()) return;
      if (!first_audio_seen) {
        result.ttfa_ms = Milliseconds(Clock::now() - run_start);
        first_audio_seen = true;
      }
      waveform.insert(waveform.end(), audio.begin(), audio.end());
    };

    const size_t generated_frames = generator.Generate(
        prompt, settings_.target_codec_frames,
        [&](const houmo::Qwen3TTSCodecFrame& frame) {
          decoder_frames.push_back(frame);
          ++frames_seen;
          const bool is_last_frame =
              frames_seen == settings_.target_codec_frames;
          if (decoder_frames.size() == PerfTts::kDecoderChunkSize ||
              is_last_frame) {
            const bool is_final = is_last_frame;
            const auto decoder_start = Clock::now();
            auto decoded = stateful_decoder_->Decode(
                decoder_frames, std::move(decoder_state), is_final);
            result.stages.stateful_decoder_ms +=
                Milliseconds(Clock::now() - decoder_start);
            ++result.stages.stateful_decoder_count;
            emit_audio(std::move(decoded.audio));
            decoder_state = std::move(decoded.state);
            decoder_frames.clear();
          }
          if (progress_callback) {
            progress_callback(frames_seen, settings_.target_codec_frames);
          }
          return true;
        });

    const auto run_end = Clock::now();
    result.e2e_ms = Milliseconds(run_end - run_start);
    result.generated_frames = generated_frames;
    result.audio_samples = waveform.size();
    result.audio_duration_s =
        static_cast<double>(result.audio_samples) / PerfTts::kSampleRate;
    result.decoder_chunks = result.stages.stateful_decoder_count;

    const auto& perf = generator.perf();
    CopyGenerationPerf(perf, &result.stages);
    result.codec_generation_ms = GenerationMilliseconds(perf);
    if (result.codec_generation_ms > 0.0) {
      result.codec_frames_per_second =
          static_cast<double>(result.generated_frames) * 1000.0 /
          result.codec_generation_ms;
    }
    result.rtf = result.e2e_ms / 1000.0 / settings_.nominal_audio_length_s;

    ValidateResult(result, first_audio_seen, decoder_frames.empty());
    const double measured_ms =
        result.stages.text_embedding_ms + result.stages.text_projection_ms +
        result.stages.prompt_prepare_ms + result.stages.talker_prefill_ms +
        result.stages.talker_decode_ms + result.stages.talker_sampling_ms +
        result.stages.codec_frame_prepare_ms +
        result.stages.code_predictor_prepare_ms +
        result.stages.code_predictor_prefill_ms +
        result.stages.code_predictor_decode_ms +
        result.stages.code_predictor_sampling_ms +
        result.stages.stateful_decoder_ms;
    result.stages.other_ms = std::max(0.0, result.e2e_ms - measured_ms);

    if (keep_waveform) result.waveform = std::move(waveform);
    return result;
  }

  const TtsPerfSettings& settings() const { return settings_; }

 private:
  void NormalizeSettings() {
    if (!std::isfinite(settings_.requested_audio_length_s) ||
        settings_.requested_audio_length_s <= 0.0) {
      throw std::invalid_argument(
          "requested_audio_length_s must be finite and greater than zero");
    }
    if (settings_.token_per_second <= 0) {
      throw std::invalid_argument(
          "token_per_second must be a positive integer");
    }

    settings_.body_text_tokens = CheckedCeil(
        static_cast<long double>(settings_.requested_audio_length_s) *
            settings_.token_per_second,
        "body text tokens");
    settings_.body_text_tokens =
        std::max<size_t>(1, settings_.body_text_tokens);
    settings_.text_projection_tokens =
        CheckedAdd(settings_.body_text_tokens, 6, "text projection tokens");
    settings_.target_codec_frames =
        StableTargetFrames(settings_.requested_audio_length_s);
    settings_.nominal_audio_length_s =
        settings_.target_codec_frames * PerfTts::kSecondsPerFrame;
    settings_.expected_audio_samples =
        CheckedMultiply(settings_.target_codec_frames,
                        PerfTts::kSamplesPerFrame, "expected audio samples");
    settings_.decoder_chunks =
        CheckedAdd(settings_.target_codec_frames,
                   PerfTts::kDecoderChunkSize - 1, "decoder chunks") /
        PerfTts::kDecoderChunkSize;
  }

  void ValidatePathsAndFixedSettings() const {
    if (settings_.device_id < 0) {
      throw std::invalid_argument("device_id must be non-negative");
    }
    if (settings_.language != "Chinese" || settings_.speaker != "vivian" ||
        settings_.seed != PerfTts::kSeed) {
      throw std::invalid_argument(
          "TTS perf fixes language=Chinese, speaker=vivian, and seed=42");
    }
    RequireFile("text_projection_path", settings_.text_projection_path);
    RequireFile("talker_prefill_path", settings_.talker_prefill_path);
    RequireFile("talker_decode_path", settings_.talker_decode_path);
    RequireFile("code_predictor_prefill_path",
                settings_.code_predictor_prefill_path);
    RequireFile("code_predictor_decode_path",
                settings_.code_predictor_decode_path);
    RequireFile("stateful_decoder_path", settings_.stateful_decoder_path);
    RequireFile("embedding_path", settings_.embedding_path);
    RequireFile("code_embedding_path", settings_.code_embedding_path);
    RequireFile("text_embedding_path", settings_.text_embedding_path);
  }

  void ValidateModelInterfaces() const {
    if (text_projection_->input_hidden_dim() !=
            houmo::Qwen3TTSTextEmbedding::kHiddenDim ||
        text_projection_->output_hidden_dim() != talker_->hidden_dim()) {
      throw std::invalid_argument(
          "TextProjection dimensions do not match text embedding and Talker");
    }
    if (code_predictor_->hidden_dim() !=
        code_predictor_embedding_->hidden_dim()) {
      throw std::invalid_argument(
          "CodePredictor embedding hidden dimension mismatch");
    }
    if (text_embedding_->vocab_size() <= kTtsEosToken) {
      throw std::invalid_argument(
          "Text embedding vocabulary does not cover fixed role/TTS tokens");
    }
    if (codec_embedding_->vocab_size() <= kVivianToken) {
      throw std::invalid_argument(
          "Talker codec embedding vocabulary does not cover fixed prompt "
          "tokens");
    }
    const size_t chunk = text_projection_->chunk_length();
    if (chunk == 0 || kRoleTokens.size() % chunk != 0 || 3 % chunk != 0 ||
        settings_.body_text_tokens % chunk != 0) {
      throw std::invalid_argument(
          "Role, body, and special token lengths must fit TextProjection "
          "chunks");
    }
    if (talker_->hidden_dim() >
        static_cast<size_t>(std::numeric_limits<int>::max())) {
      throw std::invalid_argument("Talker hidden dimension is out of range");
    }
  }

  std::vector<houmo::Token> GenerateBodyTokens() const {
    std::mt19937 text_rng(PerfTts::kSeed);
    std::uniform_int_distribution<int> distribution(
        0, text_embedding_->vocab_size() - 1);
    std::vector<houmo::Token> tokens;
    tokens.reserve(settings_.body_text_tokens);
    while (tokens.size() < settings_.body_text_tokens) {
      const houmo::Token token = distribution(text_rng);
      if (std::find(kExcludedTextTokens.begin(), kExcludedTextTokens.end(),
                    token) == kExcludedTextTokens.end()) {
        tokens.push_back(token);
      }
    }
    return tokens;
  }

  static void CopyGenerationPerf(const houmo::Qwen3TTSGenerationPerf& source,
                                 TtsStagePerf* target) {
    target->talker_prefill_ms = source.talker_prefill_seconds * 1000.0;
    target->talker_prefill_count = source.talker_prefill_count;
    target->talker_decode_ms = source.talker_decode_seconds * 1000.0;
    target->talker_decode_count = source.talker_decode_count;
    target->talker_sampling_ms = source.talker_sampling_seconds * 1000.0;
    target->talker_sampling_count = source.talker_sampling_count;
    target->codec_frame_prepare_ms = source.frame_prepare_seconds * 1000.0;
    target->code_predictor_prepare_ms =
        source.predictor_prepare_seconds * 1000.0;
    target->code_predictor_prefill_ms =
        source.predictor_prefill_seconds * 1000.0;
    target->code_predictor_prefill_count = source.predictor_prefill_count;
    target->code_predictor_decode_ms = source.predictor_decode_seconds * 1000.0;
    target->code_predictor_decode_count = source.predictor_decode_count;
    target->code_predictor_sampling_ms =
        source.predictor_sampling_seconds * 1000.0;
    target->code_predictor_sampling_count = source.predictor_sampling_count;
  }

  static double GenerationMilliseconds(
      const houmo::Qwen3TTSGenerationPerf& perf) {
    return 1000.0 *
           (perf.talker_prefill_seconds + perf.talker_decode_seconds +
            perf.talker_sampling_seconds + perf.frame_prepare_seconds +
            perf.predictor_prepare_seconds + perf.predictor_prefill_seconds +
            perf.predictor_decode_seconds + perf.predictor_sampling_seconds);
  }

  void ValidateResult(const TtsPerfResult& result, bool first_audio_seen,
                      bool decoder_buffer_empty) const {
    if (result.generated_frames != settings_.target_codec_frames) {
      throw std::runtime_error(
          "Generated codec frame count does not match target");
    }
    if (!decoder_buffer_empty ||
        result.decoder_chunks != settings_.decoder_chunks) {
      throw std::runtime_error(
          "StatefulDecoder chunk count does not match target");
    }
    if (result.audio_samples != settings_.expected_audio_samples) {
      std::ostringstream message;
      message << "Decoded sample count mismatch: expected="
              << settings_.expected_audio_samples
              << ", actual=" << result.audio_samples;
      throw std::runtime_error(message.str());
    }
    if (!first_audio_seen) {
      throw std::runtime_error("StatefulDecoder produced no audio");
    }

    const size_t frames = settings_.target_codec_frames;
    if (result.stages.talker_prefill_count != 1 ||
        result.stages.talker_decode_count != frames - 1 ||
        result.stages.talker_sampling_count != frames ||
        result.stages.code_predictor_prefill_count != frames ||
        result.stages.code_predictor_decode_count != 14 * frames ||
        result.stages.code_predictor_sampling_count != 15 * frames) {
      throw std::runtime_error(
          "TTS model call counts do not match fixed-frame mode");
    }
  }

  TtsPerfSettings settings_;
  std::vector<houmo::Token> body_tokens_;
  std::unique_ptr<houmo::Qwen3TTSTextEmbedding> text_embedding_;
  std::unique_ptr<houmo::Qwen3TTSTextProjection> text_projection_;
  std::unique_ptr<houmo::Qwen3TTSTalker> talker_;
  std::unique_ptr<houmo::Qwen3TTSCodecEmbedding> codec_embedding_;
  std::unique_ptr<houmo::Qwen3TTSCodePredictor> code_predictor_;
  std::unique_ptr<houmo::Qwen3TTSCodePredictorEmbedding>
      code_predictor_embedding_;
  std::unique_ptr<houmo::Qwen3TTSStatefulDecoder> stateful_decoder_;
  houmo::Qwen3TTSStreamingPromptBuilder prompt_builder_;
};

PerfTts::PerfTts(const TtsPerfSettings& settings)
    : impl_(std::make_unique<Impl>(settings)) {}

PerfTts::~PerfTts() = default;
PerfTts::PerfTts(PerfTts&&) noexcept = default;
PerfTts& PerfTts::operator=(PerfTts&&) noexcept = default;

void PerfTts::ValidateWorkload() const { impl_->ValidateWorkload(); }

TtsPerfResult PerfTts::Run(bool keep_waveform,
                           const TtsProgressCallback& progress_callback) {
  return impl_->Run(keep_waveform, progress_callback);
}

const TtsPerfSettings& PerfTts::settings() const { return impl_->settings(); }
