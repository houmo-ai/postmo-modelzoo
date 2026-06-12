/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: audio_processor.h
 * Description:
 *   Audio processor for ASR models - supports Whisper, GLM-ASR, Qwen3-ASR
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

#ifndef HOUMO_AUDIO_PROCESSOR_H
#define HOUMO_AUDIO_PROCESSOR_H

#include <memory>
#include <string>
#include <vector>

#include "base/houmo.h"

namespace houmo {

/**
 * @brief 音频数据
 *
 * 存储 PCM 音频数据和相关元信息。
 */
struct AudioData {
  std::vector<float> pcm;   ///< PCM 数据 (float32, 16kHz, mono)
  int sample_rate = 16000;  ///< 采样率 (Hz)
  float duration = 0.0f;    ///< 时长 (秒)
};

/**
 * @brief 音频特征
 *
 * 存储音频特征数据和维度信息。
 */
struct MelFeatures {
  std::vector<float16> data;  ///< 特征数据 (FP16)
  int feature_dim = 0;        ///< 特征维度 (80 or 128 for Mel Spectrogram)
  int num_frames = 0;         ///< 帧数
};

/**
 * @brief 音频处理器配置
 *
 * 通过配置支持不同的 ASR 模型:
 * - Whisper, GLM-ASR, Qwen3-ASR: Mel Spectrogram
 */
struct AudioProcessorConfig {
  // ========== 基础参数 ==========

  int sample_rate = 16000;  ///< 目标采样率 (Hz), 默认 16kHz

  // ========== Mel Spectrogram 参数 (Whisper/GLM-ASR/Qwen3-ASR) ==========

  int n_mels = 80;            ///< Mel bins 数量 (80 or 128 for whisper-large-v3-turbo)
  int chunk_seconds = 30;     ///< PCM 分块大小 (秒), 用于音频分段
  int encoder_window_seconds = 30;  ///< Encoder 输入窗口大小 (秒), Whisper 固定 30 秒

  // ========== 高级参数 (可选) ==========

  int fft_size = 400;         ///< FFT 大小 (25ms at 16kHz)
  int hop_length = 160;       ///< 步长 (10ms at 16kHz)
  int win_length = 400;       ///< 窗长 (25ms at 16kHz)
  float mel_fmin = 0.0f;      ///< Mel 滤波器最小频率
  float mel_fmax = 8000.0f;   ///< Mel 滤波器最大频率
};

/**
 * @brief 音频处理器
 *
 * 支持 ASR 模型的音频预处理:
 * - Whisper (Mel Spectrogram)
 * - GLM-ASR (Mel Spectrogram)
 * - Qwen3-ASR (Mel Spectrogram)
 *
 * 典型流程:
 * 1. LoadAudio() 加载音频文件
 * 2. ChunkPCM() PCM 级别分块 (30秒)
 * 3. ExtractFeatures() 提取特征
 *
 * 或使用 Process() 一站式处理。
 *
 * Reference:
 * - Whisper: /hmdd/imodelzoo/models/asr/whisper/demo.py
 * - GLM-ASR: /hmdd/imodelzoo/models/asr/glm-asr/demo.py
 * - Qwen3-ASR: /hmdd/imodelzoo/models/asr/qwen3-asr/demo_asr.py
 */
class AudioProcessor {
 public:
  /**
   * @brief 默认构造函数 (Mel Spectrogram 模式)
   */
  AudioProcessor();

  /**
   * @brief 配置构造函数
   * @param config 处理器配置
   */
  explicit AudioProcessor(const AudioProcessorConfig& config);

  // Non-copyable
  AudioProcessor(const AudioProcessor&) = delete;
  AudioProcessor& operator=(const AudioProcessor&) = delete;

  // Moveable
  AudioProcessor(AudioProcessor&&) noexcept = default;
  AudioProcessor& operator=(AudioProcessor&&) noexcept = default;

  ~AudioProcessor();

  // ========== 音频加载 ==========

  /**
   * @brief 加载音频文件
   * @param path 音频文件路径 (支持 wav, mp3, flac 等)
   * @return AudioData PCM 数据
   *
   * 内部流程:
   * - 使用 miniaudio 读取音频文件
   * - 自动重采样到 16kHz
   * - 转换为单声道
   * - 归一化到 [-1, 1] 范围
   *
   * TODO: 实现音频加载逻辑
   */
  AudioData LoadAudio(const std::string& path);

  // ========== 特征提取 ==========

  /**
   * @brief 提取音频特征
   * @param audio 输入音频
   * @return MelFeatures 音频特征
   *
   * Mel Spectrogram 模式 (Whisper/GLM-ASR/Qwen3-ASR):
   * - STFT (短时傅里叶变换)
   * - Mel Filter Bank
   * - Log 压缩
   * - 输出维度: [n_mels, n_frames]
   */
  MelFeatures ExtractFeatures(const AudioData& audio);

  // ========== 分块处理 ==========

  /**
   * @brief PCM 级别分块
   * @param audio 输入音频
   * @return 音频块列表
   *
   * 将长音频分割为固定长度 (默认 30 秒) 的块:
   * - 不足 30 秒的块用零填充
   * - 返回的每个块都是独立的 AudioData
   *
   * TODO: 实现分块逻辑
   */
  std::vector<AudioData> ChunkPCM(const AudioData& audio);

  // ========== 一站式处理 ==========

  /**
   * @brief 加载音频并提取特征 (支持自动分块)
   * @param path 音频文件路径
   * @return 特征块列表
   *
   * 完整流程:
   * 1. LoadAudio(path)
   * 2. ChunkPCM(audio)
   * 3. ExtractFeatures(chunk) for each chunk
   *
   * TODO: 实现一站式处理逻辑
   */
  std::vector<MelFeatures> Process(const std::string& path);

  // ========== 信息获取 ==========

  /**
   * @brief 获取特征维度
   * @return 特征维度 (80 or 128 for Mel Spectrogram)
   */
  int feature_dim() const;

  /**
   * @brief 获取采样率
   */
  int sample_rate() const { return config_.sample_rate; }

  /**
   * @brief 获取 Mel bins 数量
   */
  int n_mels() const { return config_.n_mels; }

  /**
   * @brief 获取配置
   */
  const AudioProcessorConfig& config() const { return config_; }

 private:
  AudioProcessorConfig config_;

  // ========== 辅助方法 ==========

  void DownmixToMono(const std::vector<float>& interleaved, int channels,
                     std::vector<float>* mono);

  bool ResampleAudio(const std::vector<float>& input, uint32_t input_sr,
                     std::vector<float>* output);

  MelFeatures ComputeMelSpectrogram(const AudioData& audio);
};

}  // namespace houmo

#endif  // HOUMO_AUDIO_PROCESSOR_H
