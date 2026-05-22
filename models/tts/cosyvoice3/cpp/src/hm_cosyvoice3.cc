/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: hm_cosyvoice3.cc
 * Description:
 *   Main CosyVoice3 TTS Pipeline implementation.
 *   Integrates all modules for end-to-end text-to-speech synthesis.
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

#include "hm_cosyvoice3.h"

#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

namespace houmo {

namespace {

bool FileExists(const std::string& path) {
  std::ifstream file(path, std::ios::binary);
  return file.good();
}

std::string ResolveEmbeddingPath(const CosyVoice3Config& config,
                                 const std::string& emb_name) {
  if (!config.embedding_dir.empty()) {
    const std::string hemm_path =
        config.embedding_dir + "/" + emb_name + ".hemm";
    if (FileExists(hemm_path)) {
      return hemm_path;
    }
    const std::string bin_path = config.embedding_dir + "/" + emb_name + ".bin";
    if (FileExists(bin_path)) {
      return bin_path;
    }
  }
  return config.GetEmbeddingPath(emb_name);
}

void AccumulatePerf(CosyVoice3Perf* total, const CosyVoice3Perf& utterance) {
  if (total == nullptr) {
    return;
  }

  total->llm_total_ms += utterance.llm_total_ms;
  total->prefill_ms += utterance.prefill_ms;
  total->prefill_tokens += utterance.prefill_tokens;
  total->ttft_ms += utterance.ttft_ms;
  total->decode_ms += utterance.decode_ms;
  total->decode_tokens += utterance.decode_tokens;
  total->speaker_emb_ms += utterance.speaker_emb_ms;
  total->speech_tokenizer_ms += utterance.speech_tokenizer_ms;
  total->flow_total_ms += utterance.flow_total_ms;
  total->flow_encoder_ms += utterance.flow_encoder_ms;
  total->flow_decoder_ms += utterance.flow_decoder_ms;
  total->vocoder_ms += utterance.vocoder_ms;
}

void FinalizePerf(CosyVoice3Perf* perf, const std::vector<float>& audio,
                  int sample_rate) {
  if (perf == nullptr) {
    return;
  }

  perf->audio_duration_s =
      sample_rate > 0 ? static_cast<float>(audio.size()) / sample_rate : 0.0f;
  perf->e2e_latency_s =
      (perf->llm_total_ms + perf->flow_total_ms + perf->vocoder_ms) / 1000.0f;
  perf->rtf = perf->audio_duration_s > 0.0f
                  ? perf->e2e_latency_s / perf->audio_duration_s
                  : 0.0f;
}

}  // namespace

// ============================================================================
// Construction & Destruction
// ============================================================================

CosyVoice3::CosyVoice3(const CosyVoice3Config& config)
    : config_(config),
      sample_rate_(config.sample_rate),
      output_dir_(config.output_dir),
      perf_history_() {
  text_frontend_ = std::make_shared<HmTextFrontend>(config.tokenizer_dir);
  if (!text_frontend_->IsLoaded()) {
    std::cerr << "ERROR: Failed to initialize text frontend!\n";
  }

  audio_processor_ = std::make_shared<HmAudio>();

  speech_tokenizer_ = std::make_shared<HmSpeechTokenizer>(
      config.GetModelPath("speech_tokenizer"), audio_processor_);
  speaker_embedding_ =
      std::make_shared<HmSpeakerEmbedding>(config.GetModelPath("campplus"));

  llm_ = std::make_shared<HmNativeLLM>(
      config.GetModelPath("llm_decoder"),
      config.GetModelPath("llm_qwen2_prefill"),
      config.GetModelPath("llm_qwen2_decode"),
      ResolveEmbeddingPath(config, "quant_embedding"),
      ResolveEmbeddingPath(config, "llm_speech_embedding"),
      ResolveEmbeddingPath(config, "llm_sos_embedding"),
      ResolveEmbeddingPath(config, "llm_task_id_embedding"));

  flow_ = std::make_shared<HmNativeFlow>(
      ResolveEmbeddingPath(config, "flow_input_embedding"),
      config.GetModelPath("flow_encoder"), config.GetModelPath("flow_spk"),
      config.GetModelPath("flow_decoder"));

  vocoder_ = std::make_shared<HmNativeVocoder>(
      config.GetModelPath("hift_part1"), config.GetModelPath("hift_part2"));
}

CosyVoice3::~CosyVoice3() = default;

CosyVoice3::CosyVoice3(CosyVoice3&& other) noexcept
    : text_frontend_(std::move(other.text_frontend_)),
      audio_processor_(std::move(other.audio_processor_)),
      speech_tokenizer_(std::move(other.speech_tokenizer_)),
      speaker_embedding_(std::move(other.speaker_embedding_)),
      config_(std::move(other.config_)),
      sample_rate_(other.sample_rate_),
      output_dir_(std::move(other.output_dir_)),
      perf_history_(std::move(other.perf_history_)) {}

CosyVoice3& CosyVoice3::operator=(CosyVoice3&& other) noexcept {
  if (this != &other) {
    config_ = std::move(other.config_);
    sample_rate_ = other.sample_rate_;
    output_dir_ = std::move(other.output_dir_);
    perf_history_ = std::move(other.perf_history_);
    text_frontend_ = std::move(other.text_frontend_);
    audio_processor_ = std::move(other.audio_processor_);
    speech_tokenizer_ = std::move(other.speech_tokenizer_);
    speaker_embedding_ = std::move(other.speaker_embedding_);
  }
  return *this;
}

// ============================================================================
// Inference Methods
// ============================================================================

std::vector<float> CosyVoice3::InferenceZeroShot(
    const std::string& tts_text, const std::string& prompt_text,
    const std::string& prompt_wav_path, float speed, CosyVoice3Perf* perf) {
  std::cout << "\n=== Zero-Shot Voice Cloning ===\n";
  std::cout << "TTS Text: " << tts_text << "...\n";
  std::cout << "Prompt Text: " << prompt_text << "...\n";
  std::cout << "Prompt WAV: " << prompt_wav_path << "\n";
  std::cout << "Speed: " << speed << "\n";

  // Check for <|endofprompt|> marker in CosyVoice3
  if (prompt_text.find("<|endofprompt|>") == std::string::npos &&
      tts_text.find("<|endofprompt|>") == std::string::npos) {
    std::cout
        << "WARNING: <|endofprompt|> not found in CosyVoice3 input text.\n";
  }

  // Normalize prompt text (split=false)
  auto prompt_text_normalized =
      text_frontend_->TextNormalize(prompt_text, false);
  std::string prompt_text_str =
      prompt_text_normalized.empty() ? prompt_text : prompt_text_normalized[0];

  // Normalize TTS text and split into utterances
  auto tts_utterances = text_frontend_->TextNormalize(tts_text, true);
  if (tts_utterances.empty()) {
    tts_utterances.push_back(tts_text);
  }

  std::vector<float> final_audio;
  CosyVoice3Perf local_perf;

  for (size_t i = 0; i < tts_utterances.size(); ++i) {
    const auto& utterance = tts_utterances[i];

    // Check for short utterance warning
    if (utterance.length() < 0.5 * prompt_text_str.length()) {
      std::cout << "WARNING: Synthesis text is shorter than prompt text.\n";
      std::cout << "  This may lead to bad performance.\n";
    }

    CosyVoice3Perf utterance_perf;
    auto frontend_input = FrontendZeroShot(
        utterance, prompt_text_str, prompt_wav_path, "", &utterance_perf);
    std::vector<float> utterance_audio =
        SynthesizeUtterance(frontend_input, speed, &utterance_perf);

    final_audio.insert(final_audio.end(), utterance_audio.begin(),
                       utterance_audio.end());
    AccumulatePerf(&local_perf, utterance_perf);
  }

  FinalizePerf(&local_perf, final_audio, sample_rate_);

  perf_history_.push_back(local_perf);
  perf_monitor_.Record(local_perf);
  if (perf) {
    *perf = local_perf;
  }

  std::cout << "\n=== Zero-Shot Complete ===\n";
  std::cout << "Total audio samples: " << final_audio.size() << "\n";
  std::cout << "Audio duration: " << local_perf.audio_duration_s
            << " seconds\n";
  std::cout << "RTF: " << local_perf.rtf << "\n";

  return final_audio;
}

std::vector<float> CosyVoice3::InferenceCrossLingual(
    const std::string& tts_text, const std::string& prompt_wav_path,
    float speed, CosyVoice3Perf* perf) {
  std::cout << "\n=== Cross-Lingual Synthesis ===\n";
  std::cout << "TTS Text: " << tts_text << "...\n";
  std::cout << "Prompt WAV: " << prompt_wav_path << "\n";
  std::cout << "Speed: " << speed << "\n";

  auto tts_utterances = text_frontend_->TextNormalize(tts_text, true);
  if (tts_utterances.empty()) {
    tts_utterances.push_back(tts_text);
  }

  std::vector<float> final_audio;
  CosyVoice3Perf local_perf;

  for (size_t i = 0; i < tts_utterances.size(); ++i) {
    const auto& utterance = tts_utterances[i];

    CosyVoice3Perf utterance_perf;
    auto frontend_input =
        FrontendCrossLingual(utterance, prompt_wav_path, "", &utterance_perf);
    std::vector<float> utterance_audio =
        SynthesizeUtterance(frontend_input, speed, &utterance_perf);

    final_audio.insert(final_audio.end(), utterance_audio.begin(),
                       utterance_audio.end());
    AccumulatePerf(&local_perf, utterance_perf);
  }

  FinalizePerf(&local_perf, final_audio, sample_rate_);
  perf_history_.push_back(local_perf);
  perf_monitor_.Record(local_perf);
  if (perf) {
    *perf = local_perf;
  }

  return final_audio;
}

std::vector<float> CosyVoice3::InferenceInstruct2(
    const std::string& tts_text, const std::string& instruct_text,
    const std::string& prompt_wav_path, float speed, CosyVoice3Perf* perf) {
  std::cout << "\n=== Instruct2 Synthesis ===\n";
  std::cout << "TTS Text: " << tts_text << "...\n";
  std::cout << "Instruction Text: " << instruct_text << "...\n";
  std::cout << "Prompt WAV: " << prompt_wav_path << "\n";
  std::cout << "Speed: " << speed << "\n";

  auto tts_utterances = text_frontend_->TextNormalize(tts_text, true);
  if (tts_utterances.empty()) {
    tts_utterances.push_back(tts_text);
  }

  std::vector<float> final_audio;
  CosyVoice3Perf local_perf;

  for (size_t i = 0; i < tts_utterances.size(); ++i) {
    const auto& utterance = tts_utterances[i];

    CosyVoice3Perf utterance_perf;
    auto frontend_input = FrontendInstruct2(
        utterance, instruct_text, prompt_wav_path, "", &utterance_perf);
    std::vector<float> utterance_audio =
        SynthesizeUtterance(frontend_input, speed, &utterance_perf);

    final_audio.insert(final_audio.end(), utterance_audio.begin(),
                       utterance_audio.end());
    AccumulatePerf(&local_perf, utterance_perf);
  }

  FinalizePerf(&local_perf, final_audio, sample_rate_);
  perf_history_.push_back(local_perf);
  perf_monitor_.Record(local_perf);
  if (perf) {
    *perf = local_perf;
  }

  return final_audio;
}

// ============================================================================
// Audio Output
// ============================================================================

bool CosyVoice3::SaveWav(const std::vector<float>& pcm_data,
                         const std::string& output_path, int bits_per_sample) {
  std::ofstream file(output_path, std::ios::binary);
  if (!file.is_open()) {
    std::cerr << "ERROR: Cannot open file for writing: " << output_path << "\n";
    return false;
  }

  // Create WAV header
  WavHeader header =
      CreateWavHeader(pcm_data.size(), sample_rate_, bits_per_sample);

  // Convert float to int16 or int32
  std::vector<int16_t> pcm_int16;
  std::vector<int32_t> pcm_int32;

  if (bits_per_sample == 16) {
    pcm_int16.resize(pcm_data.size());
    for (size_t i = 0; i < pcm_data.size(); ++i) {
      // Clamp and convert
      float sample = std::max(-1.0f, std::min(1.0f, pcm_data[i]));
      pcm_int16[i] = static_cast<int16_t>(sample * 32767.0f);
    }
    header.data_size = pcm_int16.size() * sizeof(int16_t);
    header.file_size = sizeof(WavHeader) - 8 + header.data_size;
  } else if (bits_per_sample == 32) {
    pcm_int32.resize(pcm_data.size());
    for (size_t i = 0; i < pcm_data.size(); ++i) {
      float sample = std::max(-1.0f, std::min(1.0f, pcm_data[i]));
      pcm_int32[i] = static_cast<int32_t>(sample * 2147483647.0f);
    }
    header.data_size = pcm_int32.size() * sizeof(int32_t);
    header.file_size = sizeof(WavHeader) - 8 + header.data_size;
  } else {
    std::cerr << "ERROR: Unsupported bits_per_sample: " << bits_per_sample
              << "\n";
    return false;
  }

  // Write header
  file.write(reinterpret_cast<char*>(&header), sizeof(WavHeader));

  // Write data
  if (bits_per_sample == 16) {
    file.write(reinterpret_cast<char*>(pcm_int16.data()),
               pcm_int16.size() * sizeof(int16_t));
  } else {
    file.write(reinterpret_cast<char*>(pcm_int32.data()),
               pcm_int32.size() * sizeof(int32_t));
  }

  file.close();

  std::cout << "Saved " << pcm_data.size() << " samples to " << output_path
            << "\n";
  return true;
}

bool CosyVoice3::SaveWavInt16(const std::vector<int16_t>& pcm_data_int16,
                              const std::string& output_path) {
  std::ofstream file(output_path, std::ios::binary);
  if (!file.is_open()) {
    return false;
  }

  WavHeader header = CreateWavHeader(pcm_data_int16.size(), sample_rate_, 16);
  header.data_size = pcm_data_int16.size() * sizeof(int16_t);
  header.file_size = sizeof(WavHeader) - 8 + header.data_size;

  file.write(reinterpret_cast<char*>(&header), sizeof(WavHeader));
  file.write(reinterpret_cast<const char*>(pcm_data_int16.data()),
             pcm_data_int16.size() * sizeof(int16_t));
  file.close();

  return true;
}

// ============================================================================
// Performance Reporting
// ============================================================================

void CosyVoice3::PrintPerfSummary() {
  // Use the HmPerfMonitor class for formatted reporting
  perf_monitor_.PrintSummary();

  // Also maintain backward compatibility with perf_history_ vector
  if (perf_history_.empty()) {
    return;
  }

  // Print detailed performance summary using PrintPerfSummary from
  // perf_monitor_ The perf_monitor_ already handles all the formatted output
}

bool CosyVoice3::IsReady() const {
  return text_frontend_ && text_frontend_->IsLoaded() && audio_processor_;
}

// ============================================================================
// Speaker Cache
// ============================================================================

bool CosyVoice3::AddZeroShotSpeaker(const std::string& prompt_text,
                                    const std::string& prompt_wav_path,
                                    const std::string& spk_id) {
  if (spk_id.empty()) {
    std::cerr << "ERROR: Speaker ID cannot be empty.\n";
    return false;
  }

  // Build frontend input with empty tts_text (just for caching)
  auto cached_input = FrontendZeroShot("", prompt_text, prompt_wav_path);

  // Remove text fields (we only cache prompt-related features)
  cached_input.text_tokens.clear();
  cached_input.text_len = 0;

  // Store in cache
  speaker_cache_[spk_id] = std::move(cached_input);
  return true;
}

bool CosyVoice3::HasSpeaker(const std::string& spk_id) const {
  return speaker_cache_.find(spk_id) != speaker_cache_.end();
}

const CosyVoice3FrontendInput* CosyVoice3::GetCachedSpeaker(
    const std::string& spk_id) const {
  auto it = speaker_cache_.find(spk_id);
  if (it != speaker_cache_.end()) {
    return &it->second;
  }
  return nullptr;
}

// ============================================================================
// Frontend Methods (Internal)
// ============================================================================

CosyVoice3FrontendInput CosyVoice3::FrontendZeroShot(
    const std::string& tts_text, const std::string& prompt_text,
    const std::string& prompt_wav_path, const std::string& use_cached_spk_id,
    CosyVoice3Perf* perf) {
  CosyVoice3FrontendInput input;

  // Check if we have cached speaker
  if (!use_cached_spk_id.empty() && HasSpeaker(use_cached_spk_id)) {
    const auto* cached = GetCachedSpeaker(use_cached_spk_id);

    // Copy cached features
    input.prompt_text_tokens = cached->prompt_text_tokens;
    input.prompt_text_len = cached->prompt_text_len;
    input.llm_prompt_speech_tokens = cached->llm_prompt_speech_tokens;
    input.llm_prompt_speech_token_len = cached->llm_prompt_speech_token_len;
    input.flow_prompt_speech_tokens = cached->flow_prompt_speech_tokens;
    input.flow_prompt_speech_token_len = cached->flow_prompt_speech_token_len;
    input.prompt_speech_feat = cached->prompt_speech_feat;
    input.prompt_speech_feat_len = cached->prompt_speech_feat_len;
    input.llm_embedding = cached->llm_embedding;
    input.flow_embedding = cached->flow_embedding;
    input.text_tokens = cached->text_tokens;
    input.text_len = cached->text_len;
  } else {
    // Extract features from prompt audio
    std::vector<float> pcm_16k, pcm_24k;
    std::vector<TensorType> speech_feat;
    std::vector<int> prompt_speech_tokens;
    std::vector<TensorType> speech_embedding;

    if (!ExtractPromptFeatures(prompt_wav_path, pcm_16k, pcm_24k, speech_feat,
                               prompt_speech_tokens, speech_embedding, perf)) {
      std::cerr << "ERROR: Failed to extract prompt features!\n";
      return input;
    }

    // Encode prompt text
    input.prompt_text_tokens = text_frontend_->Encode(prompt_text);
    input.prompt_text_len = static_cast<int>(input.prompt_text_tokens.size());

    // Store extracted features
    input.llm_prompt_speech_tokens = std::move(prompt_speech_tokens);
    input.llm_prompt_speech_token_len =
        static_cast<int>(input.llm_prompt_speech_tokens.size());
    input.flow_prompt_speech_tokens = input.llm_prompt_speech_tokens;
    input.flow_prompt_speech_token_len = input.llm_prompt_speech_token_len;

    input.prompt_speech_feat = std::move(speech_feat);
    input.prompt_speech_feat_len =
        static_cast<int>(input.prompt_speech_feat.size() / kMelBins);
    input.llm_embedding = std::move(speech_embedding);
    input.flow_embedding = input.llm_embedding;
  }

  // Encode target text
  input.text_tokens = text_frontend_->Encode(tts_text);
  input.text_len = static_cast<int>(input.text_tokens.size());

  return input;
}

CosyVoice3FrontendInput CosyVoice3::FrontendCrossLingual(
    const std::string& tts_text, const std::string& prompt_wav_path,
    const std::string& use_cached_spk_id, CosyVoice3Perf* perf) {
  CosyVoice3FrontendInput input =
      FrontendZeroShot(tts_text, "", prompt_wav_path, use_cached_spk_id);

  input.prompt_text_tokens.clear();
  input.prompt_text_len = 0;
  input.llm_prompt_speech_tokens.clear();
  input.llm_prompt_speech_token_len = 0;

  return input;
}

CosyVoice3FrontendInput CosyVoice3::FrontendInstruct2(
    const std::string& tts_text, const std::string& instruct_text,
    const std::string& prompt_wav_path, const std::string& use_cached_spk_id,
    CosyVoice3Perf* perf) {
  CosyVoice3FrontendInput input = FrontendZeroShot(
      tts_text, instruct_text, prompt_wav_path, use_cached_spk_id);

  input.llm_prompt_speech_tokens.clear();
  input.llm_prompt_speech_token_len = 0;

  return input;
}

// ============================================================================
// Feature Extraction Helpers
// ============================================================================

bool CosyVoice3::ExtractPromptFeatures(
    const std::string& prompt_wav_path, std::vector<float>& pcm_16k,
    std::vector<float>& pcm_24k, std::vector<TensorType>& speech_feat,
    std::vector<int>& prompt_speech_tokens,
    std::vector<TensorType>& speech_embedding, CosyVoice3Perf* perf) {
  // Load audio at 16kHz (for speech tokenizer and speaker embedding)
  if (!audio_processor_->LoadAudio(prompt_wav_path, kSampleRate16k)) {
    std::cerr << "ERROR: Failed to load prompt audio at 16kHz!\n";
    return false;
  }

  pcm_16k = audio_processor_->GetPcmData();

  // Extract speech tokens
  prompt_speech_tokens = speech_tokenizer_->Extract(pcm_16k, perf);

  // Extract speaker embedding
  speech_embedding = speaker_embedding_->Extract(pcm_16k, perf);

  // Load audio at 24kHz (for mel features)
  if (!audio_processor_->LoadAudio(prompt_wav_path, kSampleRate24k)) {
    std::cerr << "ERROR: Failed to load prompt audio at 24kHz!\n";
    return false;
  }
  pcm_24k = audio_processor_->GetPcmData();

  // Compute 80-bin mel spectrogram
  auto mel_features = audio_processor_->ComputeMelSpectrogram80(pcm_24k);
  speech_feat = std::move(mel_features.data);

  if (sample_rate_ == 24000) {
    int ml = mel_features.n_frames;
    int sl = prompt_speech_tokens.size();
    int min_len = std::min(ml / 2, sl);

    prompt_speech_tokens.resize(min_len);
    speech_feat.resize(min_len * 2 * kMelBins);
  }

  return true;
}

std::vector<float> CosyVoice3::SynthesizeUtterance(
    const CosyVoice3FrontendInput& frontend_input, float speed,
    CosyVoice3Perf* perf) {
  std::vector<int> tts_speech_tokens = llm_->Inference(frontend_input, perf);

  std::vector<float> mel_spectrogram =
      flow_->Inference(frontend_input, tts_speech_tokens, perf);

  return vocoder_->Inference(mel_spectrogram, speed, perf);
}

// ============================================================================
// WAV Helper
// ============================================================================

WavHeader CosyVoice3::CreateWavHeader(size_t num_samples, int sample_rate,
                                      int bits_per_sample) {
  WavHeader header;

  // RIFF chunk
  std::memcpy(header.riff, "RIFF", 4);
  header.file_size =
      sizeof(WavHeader) - 8 + num_samples * (bits_per_sample / 8);
  std::memcpy(header.wave, "WAVE", 4);

  // fmt chunk
  std::memcpy(header.fmt, "fmt ", 4);
  header.fmt_size = 16;
  header.audio_format = 1;  // PCM
  header.num_channels = 1;  // Mono
  header.sample_rate = sample_rate;
  header.bits_per_sample = bits_per_sample;
  header.block_align = header.num_channels * (bits_per_sample / 8);
  header.byte_rate = header.sample_rate * header.block_align;

  // data chunk
  std::memcpy(header.data, "data", 4);
  header.data_size = num_samples * (bits_per_sample / 8);

  return header;
}

// ============================================================================
// Command Line Argument Parsing
// ============================================================================

CosyVoice3Config ParseArgs(int argc, char* argv[]) {
  CosyVoice3Config config;

  // Default values
  config.tokenizer_dir = "./Fun-CosyVoice3-0.5B-2512";
  config.model_dir = "./output/xh2";
  config.embedding_dir = "./output/xh2/hmquant";
  config.output_dir = "./results-cpp";
  config.device_id = 0;
  config.sample_rate = 24000;

  // Parse arguments
  for (int i = 1; i < argc; ++i) {
    std::string arg = argv[i];
    std::string next_arg = (i + 1 < argc) ? argv[i + 1] : "";

    if (arg == "--tokenizer_dir" && !next_arg.empty()) {
      config.tokenizer_dir = next_arg;
      ++i;
    } else if (arg == "--model_dir" && !next_arg.empty()) {
      config.model_dir = next_arg;
      ++i;
    } else if (arg == "--embedding_dir" && !next_arg.empty()) {
      config.embedding_dir = next_arg;
      ++i;
    } else if (arg == "--output_dir" && !next_arg.empty()) {
      config.output_dir = next_arg;
      ++i;
    } else if (arg == "--device_id" && !next_arg.empty()) {
      config.device_id = std::stoi(next_arg);
      ++i;
    } else if (arg == "--help" || arg == "-h") {
      PrintUsage(argv[0]);
      exit(0);
    }
  }

  return config;
}

void PrintUsage(const char* program_name) {
  std::cout << "\nCosyVoice3 TTS Demo - C++ Implementation\n";
  std::cout << "\nUsage: " << program_name << " [options]\n";
  std::cout << "\nOptions:\n";
  std::cout << "  --tokenizer_dir <path>   Tokenizer directory (default: "
               "./Fun-CosyVoice3-0.5B-2512)\n";
  std::cout << "  --model_dir <path>       Model directory for .hmm files "
               "(default: ./output/xh2)\n";
  std::cout << "  --embedding_dir <path>   Embedding directory for .hemm files "
               "(default: ./3rdparty/embeddings)\n";
  std::cout << "  --output_dir <path>      Output directory for generated "
               "audio (default: ./results)\n";
  std::cout << "  --device_id <id>         Device ID for TCIM (default: 0)\n";
  std::cout << "  --help, -h               Show this help message\n";
  std::cout << "\nExample:\n";
  std::cout
      << "  " << program_name
      << " --tokenizer_dir /path/to/tokenizer --model_dir /path/to/models\n";
  std::cout << "\n";
}

}  // namespace houmo