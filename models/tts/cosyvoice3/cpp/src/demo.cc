/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: demo.cc
 * Description:
 *   Main program entry point for CosyVoice3 TTS demo.
 *   Demonstrates zero-shot voice cloning, cross-lingual generation,
 *   and instruction-following synthesis.
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

#include <sys/stat.h>   // for mkdir and stat
#include <sys/types.h>  // for mode_t

#include <iostream>
#include <string>
#include <vector>

#include "hm_cosyvoice3.h"

/**
 * @brief Run zero-shot voice cloning demo.
 */
void RunZeroShotDemo(houmo::CosyVoice3& cosyvoice,
                     const std::string& prompt_wav) {
  std::cout << "\n========================================\n";
  std::cout << "Demo 1: Zero-Shot Voice Cloning\n";
  std::cout << "========================================\n";

  // Use simple text without special tokens to avoid tokenizers-cpp crash
  // Special tokens like <|endofprompt|> and [j][ǐ] may cause issues in C++
  // tokenizers library
  std::string tts_text =
      "下面为您朗诵一段绕口令，希望您[j][ǐ]予好评，朗诵开始: "
      "八百标兵奔北坡，北坡炮兵并排跑，炮兵怕把标兵碰，标兵怕碰炮兵炮。";
  std::string prompt_text =
      "You are a helpful "
      "assistant.<|endofprompt|>希望你以后能够做的比我还好呦。";

  std::cout << "\nInput Text: " << tts_text << "\n";
  std::cout << "Prompt Text: " << prompt_text << "\n";

  // Generate audio
  houmo::CosyVoice3Perf perf;
  auto audio = cosyvoice.InferenceZeroShot(tts_text, prompt_text, prompt_wav,
                                           1.0f, &perf);

  if (audio.empty()) {
    std::cerr << "ERROR: Zero-shot demo generated empty audio output.\n";
    return;
  }

  const std::string output_path =
      cosyvoice.GetOutputDir() + "/cosyvoice3_zero_shot_0.wav";
  if (!cosyvoice.SaveWav(audio, output_path)) {
    std::cerr << "ERROR: Failed to save generated audio to " << output_path
              << "\n";
    return;
  }

  std::cout << "Saved zero-shot demo audio to: " << output_path << "\n";
  perf.Print();
}

/**
 * @brief Run cross-lingual synthesis demo.
 */
void RunCrossLingualDemo(houmo::CosyVoice3& cosyvoice,
                         const std::string& prompt_wav) {
  std::cout << "\n========================================\n";
  std::cout << "Demo 2: Cross-Lingual Synthesis\n";
  std::cout << "========================================\n";

  std::string tts_text =
      "You are a helpful assistant.<|endofprompt|>[breath]因为他们那一辈人"
      "[breath]在乡里面住的要习惯一点，[breath]邻居都很活络，[breath]嗯，"
      "都很熟悉。[breath]";

  houmo::CosyVoice3Perf perf;
  auto audio =
      cosyvoice.InferenceCrossLingual(tts_text, prompt_wav, 1.0f, &perf);

  if (audio.empty()) {
    std::cerr << "ERROR: Cross-lingual demo generated empty audio output.\n";
    return;
  }

  const std::string output_path =
      cosyvoice.GetOutputDir() + "/cosyvoice3_fine_grained_control_0.wav";
  if (!cosyvoice.SaveWav(audio, output_path)) {
    std::cerr << "ERROR: Failed to save generated audio to " << output_path
              << "\n";
    return;
  }

  std::cout << "Saved cross-lingual demo audio to: " << output_path << "\n";
  perf.Print();
}

/**
 * @brief Run instruct2 synthesis demo.
 */
void RunInstruct2Demo(houmo::CosyVoice3& cosyvoice,
                      const std::string& prompt_wav) {
  std::cout << "\n========================================\n";
  std::cout << "Demo 3: Instruct2 Synthesis\n";
  std::cout << "========================================\n";

  std::string tts_text = "好少咯，一般系放嗰啲国庆啊，中秋嗰啲可能会咯。";
  std::string instruct_text =
      "You are a helpful assistant. 请用广东话表达。<|endofprompt|>";

  houmo::CosyVoice3Perf perf;
  auto audio = cosyvoice.InferenceInstruct2(tts_text, instruct_text, prompt_wav,
                                            1.0f, &perf);

  if (audio.empty()) {
    std::cerr << "ERROR: Instruct2 demo generated empty audio output.\n";
    return;
  }

  const std::string output_path =
      cosyvoice.GetOutputDir() + "/cosyvoice3_instruct_cantonese_0.wav";
  if (!cosyvoice.SaveWav(audio, output_path)) {
    std::cerr << "ERROR: Failed to save generated audio to " << output_path
              << "\n";
    return;
  }

  std::cout << "Saved instruct2 demo audio to: " << output_path << "\n";
  perf.Print();
}

int main(int argc, char* argv[]) {
  std::cout << "\n";
  std::cout << "==============================================\n";
  std::cout << "  CosyVoice3 TTS Demo - C++ Implementation\n";
  std::cout << "  Phase 11: Complete TTS Pipeline\n";
  std::cout << "==============================================\n";
  std::cout << "\n";

  // Parse command line arguments
  houmo::CosyVoice3Config config = houmo::ParseArgs(argc, argv);

  // Create output directory if it doesn't exist
  struct stat st;
  if (stat(config.output_dir.c_str(), &st) != 0) {
    // Directory doesn't exist, create it
    if (mkdir(config.output_dir.c_str(), 0755) != 0) {
      std::cerr << "ERROR: Failed to create output directory: "
                << config.output_dir << "\n";
    } else {
      std::cout << "Created output directory: " << config.output_dir << "\n";
    }
  }

  // Print configuration
  std::cout << "\n--- Configuration ---\n";
  std::cout << "Tokenizer Dir: " << config.tokenizer_dir << "\n";
  std::cout << "Model Dir: " << config.model_dir << "\n";
  std::cout << "Embedding Dir: " << config.embedding_dir << "\n";
  std::cout << "Output Dir: " << config.output_dir << "\n";
  std::cout << "Device ID: " << config.device_id << "\n";
  std::cout << "Sample Rate: " << config.sample_rate << "\n";
  std::cout << "----------------------\n";

  // Find prompt wav file
  // Note: In actual deployment, prompt wav path would be from command line
  std::string prompt_wav = "./zero_shot_prompt.wav";
  if (prompt_wav.empty()) {
    std::cout << "\nWARNING: No prompt wav file found!\n";
    std::cout << "Please provide a prompt wav file for voice cloning.\n";
    std::cout << "Expected files: zero_shot_prompt.wav, prompt.wav, or "
                 "test_prompt.wav\n";
    std::cout << "\nProceeding with initialization only...\n";
  }

  try {
    // Initialize CosyVoice3
    std::cout << "\n--- Initializing CosyVoice3 ---\n";
    houmo::CosyVoice3 cosyvoice(config);

    // Check if ready
    if (!cosyvoice.IsReady()) {
      std::cerr << "\nERROR: CosyVoice3 initialization failed!\n";
      std::cerr << "Please check model and embedding paths.\n";
      return 1;
    }

    std::cout << "\nCosyVoice3 initialized successfully!\n";

    // Run demos if prompt wav is available
    if (!prompt_wav.empty()) {
      // Demo 1: Zero-Shot Voice Cloning
      RunZeroShotDemo(cosyvoice, prompt_wav);
      RunCrossLingualDemo(cosyvoice, prompt_wav);
      RunInstruct2Demo(cosyvoice, prompt_wav);

      // Print overall performance summary
      std::cout << "\n========================================\n";
      std::cout << "Overall Performance Summary\n";
      std::cout << "========================================\n";
      cosyvoice.PrintPerfSummary();
    } else {
      std::cout << "\nSkipping demo runs (no prompt wav).\n";
      std::cout << "To run demos, please provide a prompt wav file.\n";

      // Print module status
      std::cout << "\n--- Module Status ---\n";
      std::cout << "All modules loaded and ready for inference.\n";
      std::cout << "To run inference, call:\n";
      std::cout
          << "  - InferenceZeroShot(tts_text, prompt_text, prompt_wav_path)\n";
      std::cout << "  - InferenceCrossLingual(tts_text, prompt_wav_path)\n";
      std::cout << "  - InferenceInstruct2(tts_text, instruct_text, "
                   "prompt_wav_path)\n";
    }

    std::cout << "\n========================================\n";
    std::cout << "Demo Complete\n";
    std::cout << "========================================\n\n";

    return 0;

  } catch (const std::exception& e) {
    std::cerr << "\nERROR: Exception caught: " << e.what() << "\n";
    return 1;
  } catch (...) {
    std::cerr << "\nERROR: Unknown exception caught!\n";
    return 1;
  }
}