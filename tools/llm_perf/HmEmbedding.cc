#include "HmEmbedding.h"

HmEmbedding::HmEmbedding(const std::string &embeddingWeightPath,
                         const int &embedding_len,
                         const int &prefill_len) : prefill_length(prefill_len),
                                                   embedding_length(embedding_len)
{
  // 读取embedding.bin 额外申请1M空间 直接decode时返回embed_w对应地址的指针
  embed_w = readEmbeddingWeight<tensor_type>(embeddingWeightPath, prefill_length * embedding_length);
  if(embed_w.get() == nullptr){
    throw std::runtime_error("read embed weight failed! \n");
  }
  ptr = new tensor_type[prefill_length * embedding_length];
  if(ptr == nullptr){
    throw std::runtime_error("malloc ptr failed! \n");
  }
}

HmEmbedding::~HmEmbedding()
{
  delete ptr;
}

tensor_type *HmEmbedding::EmbeddingTokens(const std::vector<int> &ids) {
  uint64_t num_tokens = ids.size();

  if (!ids.size()) {
    return nullptr;
  }

  if (num_tokens == 1) {
    return reinterpret_cast<tensor_type *>(&embed_w[ids[0] * embedding_length]);
  }

  for (int index = 0; index < ids.size(); index++) {
    int embedWeightIndex = ids[index];
    memcpy(reinterpret_cast<void *>(&ptr[index * embedding_length]), reinterpret_cast<void *>(&embed_w[embedWeightIndex * embedding_length]), embedding_length * sizeof(tensor_type));
  }

  return ptr;
}