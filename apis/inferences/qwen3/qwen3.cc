#include "HmQwenInfer.h"
#include "Hmtokenizer.h"
#include "tcim/tcim_runtime.h"
#include <codecvt>
#include <filesystem>
#include <locale>
#include <string>
#ifdef _MSC_VER
#include <Windows.h>
#endif

int main(int argc, char *argv[]) {
#ifdef _MSC_VER
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
    // check args
    std::string prefillModelPath, decodeModelPath, tokenizerJsonPath, embeddingWeightPath;
    if (argc == 1) {
        prefillModelPath = "qwen3_prefill.hmm";
        decodeModelPath = "qwen3_decode.hmm";
        tokenizerJsonPath = "qwen3-8b/tokenizer.json";
        embeddingWeightPath = "hmquant/quant_embedding.bin";
    } else if (argc == 5) {
        prefillModelPath = argv[1];
        decodeModelPath = argv[2];
        tokenizerJsonPath = argv[3];
        embeddingWeightPath = argv[4];
    } else {
        std::cerr << "Usage:\n\r<1> : ./${demo_name} \n\r<2> : ./${demo_name} <prefillModelPath> <decodeModelPath> "
                  << " <tokenizerJsonPath> <embeddingWeightPath>"
                  << std::endl;
        return -1;
    }

    if (!std::filesystem::exists(prefillModelPath) || !std::filesystem::exists(decodeModelPath) || !std::filesystem::exists(tokenizerJsonPath) || !std::filesystem::exists(embeddingWeightPath)) {
        std::cerr << "Usage:\n  <1> : ./${demo_name} \n  <2> : ./${demo_name} <prefillModelPath> <decodeModelPath> "
                  << " <tokenizerJsonPath> <embeddingWeightPath>"
                  << std::endl;
        std::cerr << "Please Check files exists!" << std::endl;
        return -2;
    }

    // check env
    const char *houmo_target_env = getenv("HOUMO_TARGET");
    std::string houmo_target =
        houmo_target_env != nullptr ? std::string(houmo_target_env) : "houmo";
    if (houmo_target != "xh1" && houmo_target != "xh2") {
        std::cerr << "Unsupported backend " << houmo_target << std::endl;
        exit(-1);
    } else {
        std::cout << houmo_target << std::endl;
        printf("tcim version: %s, houmo_target:%s.\n",
               tcim::GetVersion().c_str(), houmo_target.c_str());
    }

    std::unique_ptr<HmQwenInfer> Qwen3Infer = std::make_unique<HmQwenInfer>(prefillModelPath,
                                                                            decodeModelPath, tokenizerJsonPath, embeddingWeightPath);
    Qwen3Infer->chat("请介绍一下存算一体技术的优势");
    Qwen3Infer.reset();

    return 0;
}