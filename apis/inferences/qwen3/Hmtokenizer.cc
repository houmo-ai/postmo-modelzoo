#include "Hmtokenizer.h"

HmTokenizer::HmTokenizer(const std::string &tokenizerJsonPath,
                         const std::string &embeddingWeightPath,
                         const int &embedding_len,
                         const int &prefill_len) : prefill_length(prefill_len),
                                                   embedding_length(embedding_len)
{
  // 读取tokenizer.json
  auto blob = LoadBytesFromFile(tokenizerJsonPath);
  tok = Tokenizer::FromBlobJSON(blob);
  // 读取embedding.bin 额外申请1M空间 直接decode时返回embed_w对应地址的指针
  embed_w = readEmbeddingWeight<half>(embeddingWeightPath, prefill_length * embedding_length);
  ptr = new half[prefill_length * embedding_length];
}

HmTokenizer::~HmTokenizer()
{
  tok.reset();
  delete ptr;
}

std::string HmTokenizer::ApplyChatTemplate(const std::vector<Message> &msgs,
                                           bool add_generation_prompt,
                                           bool enable_thinking)
{

  std::string out;
  out.reserve(1024);
  // 1. 循环 messages
  for (const auto &m : msgs)
  {
    out.append("<|im_start|>");
    out.append(m.role);
    out.push_back('\n');
    out.append(m.content);
    out.append("<|im_end|>\n");
  }
  // 2. 是否追加 assistant 提示
  if (add_generation_prompt)
  {
    out.append("<|im_start|>assistant\n");
  }

  if (!enable_thinking)
  {
    out.append("<think>\n");
    out.append("\n");
    out.append("</think>\n");
    out.append("\n");
  }

  return out;
}

std::vector<int> HmTokenizer::Encode(const std::string &text)
{
  std::vector<int> ids = tok->Encode(text);
  return ids;
}

std::string HmTokenizer::Decode(const std::vector<int32_t> &ids)
{
  return tok->Decode(ids);
}

half *HmTokenizer::EmbeddingTokens(const std::vector<int> &ids)
{
  uint64_t num_tokens = ids.size();

  if (!ids.size())
  {
    return nullptr;
  }

  if (num_tokens == 1)
  {
    return reinterpret_cast<half *>(&embed_w[ids[0] * embedding_length]);
  }

  for (int index = 0; index < num_tokens; index++)
  {
    int embedWeightIndx = ids[index];
    memcpy(reinterpret_cast<void *>(&ptr[index * embedding_length]), reinterpret_cast<void *>(&embed_w[embedWeightIndx * embedding_length]), embedding_length * sizeof(half));
  }

  return ptr;
}

half *HmTokenizer::EmbeddingTokens(const std::vector<Message> &msgs,
                                   bool add_generation_prompt,
                                   bool enable_thinking)
{
  if (!msgs.size())
  {
    return nullptr;
  }

  std::string rendered = ApplyChatTemplate(msgs, add_generation_prompt, enable_thinking);

  std::vector<int> ids = Encode(rendered);

  return EmbeddingTokens(ids);
}