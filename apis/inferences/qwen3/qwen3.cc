/*
 * Qwen3 Model Inference Demo - Demonstrates how to use the HoumoAI Qwen3 inference library
 *
 * Copyright (c) 2025 HOUMOAI
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */

#include <codecvt>
#include <filesystem>
#include <locale>
#include <string>

#include "HmQwenInfer.h"
#include "Hmtokenizer.h"
#include "tcim/tcim_runtime.h"
#ifdef _MSC_VER
#include <Windows.h>
#endif

int main(int argc, char *argv[]) {
#ifdef _MSC_VER
    SetConsoleOutputCP(CP_UTF8);  // Set console output code page to UTF-8 for Windows
    SetConsoleCP(CP_UTF8);        // Set console input code page to UTF-8 for Windows
#endif

    // Paths for model files, tokenizer, and embedding weights
    std::string prefillModelPath, decodeModelPath, tokenizerJsonPath, embeddingWeightPath;

    // Handle command line arguments
    if (argc == 1) {
        // Default paths if no arguments are provided
        prefillModelPath = "qwen3_prefill.hmm";
        decodeModelPath = "qwen3_decode.hmm";
        tokenizerJsonPath = "qwen3-8b/tokenizer.json";
        embeddingWeightPath = "hmquant/quant_embedding.bin";
    } else if (argc == 5) {
        // Use paths provided via command line arguments
        prefillModelPath = argv[1];
        decodeModelPath = argv[2];
        tokenizerJsonPath = argv[3];
        embeddingWeightPath = argv[4];
    } else {
        // Print usage information if invalid number of arguments
        std::cerr << "Usage:\n  <1> : ./${demo_name}\n  <2> : ./${demo_name} <prefillModelPath> <decodeModelPath> <tokenizerJsonPath> <embeddingWeightPath>" << std::endl;
        return -1;
    }

    // Check if all required files exist
    if (!std::filesystem::exists(prefillModelPath) ||
        !std::filesystem::exists(decodeModelPath) ||
        !std::filesystem::exists(tokenizerJsonPath) ||
        !std::filesystem::exists(embeddingWeightPath)) {
        std::cerr << "Usage:\n  <1> : ./${demo_name}\n  <2> : ./${demo_name} <prefillModelPath> <decodeModelPath> <tokenizerJsonPath> <embeddingWeightPath>" << std::endl;
        std::cerr << "Please check that all files exist!" << std::endl;
        return -2;
    }

    // Check and validate the HOUMO_TARGET environment variable
    const char *houmo_target_env = getenv("HOUMO_TARGET");
    std::string houmo_target = (houmo_target_env != nullptr) ? std::string(houmo_target_env) : "houmo";

    // Only xh2 backend is supported
    if (houmo_target != "xh2") {
        std::cerr << "Unsupported backend: " << houmo_target << std::endl;
        exit(-1);
    } else {
        // Print backend and tcim version information
        std::cout << "Backend: " << houmo_target << std::endl;
        printf("tcim version: %s, houmo_target: %s.\n", tcim::GetVersion().c_str(), houmo_target.c_str());
    }

    // Initialize the Qwen3 inference engine
    std::unique_ptr<HmQwenInfer> qwen3Infer = std::make_unique<HmQwenInfer>(prefillModelPath, decodeModelPath, tokenizerJsonPath, embeddingWeightPath);

    // Run chat inference with a sample question
    qwen3Infer->Chat("请介绍一下存算一体技术的优势");

    // Clean up the inference engine
    qwen3Infer.reset();

    return 0;
}