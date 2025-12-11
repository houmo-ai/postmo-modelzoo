#ifndef __TOKENIZER_H__
#define __TOKENIZER_H__

#include "utils.h"

using half_float::half;
using tokenizers::Tokenizer;
using tensor_type = half;

/**
  class HmTokenizer
 */
class HmTokenizer {
 public:
  HmTokenizer(const std::string &tokenizerJsonPath,
              const std::string &embeddingWeightPath, const int &embedding_len,
              const int &prefill_len);
  HmTokenizer(const HmTokenizer &it) = delete;
  HmTokenizer &operator=(const HmTokenizer &it) = delete;
  HmTokenizer(HmTokenizer &&it) noexcept = default;
  HmTokenizer &operator=(HmTokenizer &&it) noexcept = default;
  ~HmTokenizer();

  // 分步完成
  std::string ApplyChatTemplate(const std::vector<Message> &msgs,
                                bool add_generation_prompt = true,
                                bool enable_thinking = false);
  std::vector<int> Encode(const std::string &text);
  std::string Decode(const std::vector<int32_t> &ids);
  tensor_type *EmbeddingTokens(const std::vector<int> &ids);

  // 一步完成
  tensor_type *EmbeddingTokens(const std::vector<Message> &msgs,
                               bool add_generation_prompt = true,
                               bool enable_thinking = false);

 private:
  // 存储读取tokenizer.json后的分词器
  std::unique_ptr<Tokenizer> tok;
  // embedding_weight.pt
  std::unique_ptr<tensor_type[]> embed_w;

  tensor_type *ptr = nullptr;
  size_t ptr_size = 0;
  // 属性
  int prefill_length = 0;
  int embedding_length = 0;
};

#endif  // __TOKENIZER_H__