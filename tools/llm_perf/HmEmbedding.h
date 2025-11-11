#ifndef __EMBEDDING_H__
#define __EMBEDDING_H__

#include <string>
#include <vector>
#include <unordered_map>
#include <memory>
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>
#include <algorithm>
#include <locale>
#include <codecvt>
#include "half.hpp"
#include "utils.h"

#ifdef BACKEND_XH1
using tensor_type = int16_t;
#else
using tensor_type = half_float::half;
#endif

/**
 * 读取文件
 * @param path              文件路径
 * @param n_elems_align     读取完文件后结尾补充空元素个数，默认0
 * @return                  成功返回unique_ptr，否则 nullptr
 */
template <typename T>
std::unique_ptr<T[]> readEmbeddingWeight(const std::string &path,
                                         size_t n_elems_align = 0)
{
  std::ifstream ifs(path, std::ios::binary);
  if (!ifs)
  {
    throw std::runtime_error("invalid embedding weight file!");
  }

  ifs.seekg(0, std::ios::end);
  const std::size_t n_bytes = ifs.tellg();
  ifs.seekg(0);

  const std::size_t n_elem = n_bytes / sizeof(T) + n_elems_align;
  auto ptr = std::make_unique<T[]>(n_elem);
  ifs.read(reinterpret_cast<char *>(ptr.get()), n_bytes);
  ifs.close();
  memset(reinterpret_cast<char *>(ptr.get()) + n_bytes, 0, n_elems_align * sizeof(T));
  return ptr;
}

class HmEmbedding
{
public:
  HmEmbedding(const std::string &embeddingWeightPath,
              const int &embedding_len,
              const int &prefill_len);
  HmEmbedding(const HmEmbedding &it) = delete;
  HmEmbedding &operator=(const HmEmbedding &it) = delete;
  HmEmbedding(HmEmbedding &&it) noexcept = default;
  HmEmbedding &operator=(HmEmbedding &&it) noexcept = default;
  ~HmEmbedding();

  tensor_type *EmbeddingTokens(const std::vector<int> &ids);


private:
  // embedding_weight.pt
  std::unique_ptr<tensor_type[]> embed_w;
  tensor_type *ptr = nullptr;
  size_t ptr_size = 0;
  // 属性
  int prefill_length = 0;
  int embedding_length = 0;
};

#endif // __EMBEDDING_H__