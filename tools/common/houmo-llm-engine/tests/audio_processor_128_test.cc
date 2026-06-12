/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: audio_processor_128_test.cc
 * Description:
 *   Test AudioProcessor with 128 mel bins (for whisper-large-v3-turbo validation)
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

#include <iostream>
#include <vector>
#include <cmath>
#include <filesystem>

#include "modules/audio_processor.h"

int main() {
    // 使用 128 mel bins (与 whisper-large-v3-turbo 对齐)
    houmo::AudioProcessorConfig config;
    config.n_mels = 128;
    config.chunk_seconds = 30;

    houmo::AudioProcessor processor(config);

    std::string audio_path = "../tests/data/audio.mp3";
    auto audio = processor.LoadAudio(audio_path);
    if (audio.pcm.empty()) {
        std::cerr << "Failed to load audio" << std::endl;
        return 1;
    }

    std::cout << "Audio loaded: duration=" << audio.duration << "s, samples=" << audio.pcm.size() << std::endl;

    auto features = processor.ExtractFeatures(audio);
    std::cout << "Features: dim=" << features.feature_dim << ", frames=" << features.num_frames << std::endl;
    std::cout << "Features size: " << features.data.size() << std::endl;

    // 打印前 10 个值
    std::cout << "C++ first 10 values: ";
    for (int i = 0; i < 10 && i < static_cast<int>(features.data.size()); ++i) {
        std::cout << static_cast<float>(features.data[i]) << " ";
    }
    std::cout << std::endl;

    // 打印最后 10 个值
    std::cout << "C++ last 10 values: ";
    int start = features.data.size() - 10;
    for (int i = start; i < static_cast<int>(features.data.size()); ++i) {
        std::cout << static_cast<float>(features.data[i]) << " ";
    }
    std::cout << std::endl;

    // 计算统计信息
    float min_val = 1e10, max_val = -1e10, sum = 0;
    for (auto& v : features.data) {
        float val = static_cast<float>(v);
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    std::cout << "C++ min: " << min_val << ", max: " << max_val << ", mean: " << sum / features.data.size() << std::endl;

    return 0;
}
