#ifndef __UTILS_H__
#define __UTILS_H__

#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include <fstream>
#include <cassert>
#include <tokenizers_cpp.h>
#include <half.hpp>
#include <locale>
#include <codecvt>
#include <eigen3/unsupported/Eigen/CXX11/Tensor>

using half_float::half;

struct Message
{
  std::string role;
  std::string content;
};

static std::string
LoadBytesFromFile(const std::string &path)
{
  std::ifstream fs(path, std::ios::in | std::ios::binary);
  if (fs.fail())
  {
    std::cerr << "Cannot open " << path << std::endl;
    exit(1);
  }
  std::string data;
  fs.seekg(0, std::ios::end);
  size_t size = static_cast<size_t>(fs.tellg());
  fs.seekg(0, std::ios::beg);
  data.resize(size);
  fs.read(data.data(), size);
  return data;
}

static void PrintEncodeResult(const std::vector<int> &ids)
{
  std::cout << "tokens=[";
  for (size_t i = 0; i < ids.size(); ++i)
  {
    if (i != 0)
      std::cout << ", ";
    std::cout << ids[i];
  }
  std::cout << "]" << std::endl;
}

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
    return nullptr; // 文件打开失败
  }

  ifs.seekg(0, std::ios::end);
  const std::size_t n_bytes = ifs.tellg();
  ifs.seekg(0);

  const std::size_t n_elem = n_bytes / sizeof(T) + n_elems_align;
  auto ptr = std::make_unique<T[]>(n_elem); // 自动 delete[]
  ifs.read(reinterpret_cast<char *>(ptr.get()), n_bytes);
  ifs.close();
  return ptr;
}

/**
 * 将数据写回 .bin 文件
 * @param path              输出文件完整路径
 * @param embedding_weights 数组首地址
 * @param n                 元素个数
 * @return                  成功返回 true，否则 false
 */
template <typename T>
static bool writeEmbeddingWeight(const std::string &path,
                                 const T *embedding_weights,
                                 size_t n)
{
  if (!embedding_weights || n == 0)
    return false;

  std::ofstream ofs(path, std::ios::binary);
  if (!ofs)
    return false;

  ofs.write(reinterpret_cast<const char *>(embedding_weights),
            n * sizeof(T));
  return ofs.good();
}

/**
 * 计算utf8字符串的u32编码长度
 * @param u8                std::string utf8编码
 * @return                  返回u32string的length
 */
static std::size_t utf8_len(std::string_view u8)
{
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.from_bytes(u8.data(), u8.data() + u8.size()).size();
}

/**
 * utf8字符串转u32string
 * @param u8                std::string utf8编码
 * @return                  返回u32string
 */
static std::u32string utf8_to_u32(const std::string &u8)
{
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.from_bytes(u8);
}

/**
 * u32string转utf8字符串
 * @param u32               std::u32string u32编码
 * @return                  返回std::string
 */
static std::string u32_to_utf8(const std::u32string &u32)
{
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> conv;
  return conv.to_bytes(u32);
}

static bool is_valid_char(char32_t cp) noexcept
{
  return
      // CJK 统一表意符号
      (cp >= 0x4E00u && cp <= 0x9FFFu) ||
      (cp >= 0x3400u && cp <= 0x4DBFu) ||
      (cp >= 0x20000u && cp <= 0x2A6DFu) ||
      (cp >= 0x2A700u && cp <= 0x2B73Fu) ||
      (cp >= 0x2B740u && cp <= 0x2B81Fu) ||
      (cp >= 0x2B820u && cp <= 0x2CEAFu) ||
      // 兼容区
      (cp >= 0xF900u && cp <= 0xFAFFu) ||
      (cp >= 0x2F800u && cp <= 0x2FA1Fu) ||
      // ASCII 字母
      (cp >= 0x0041u && cp <= 0x005Au) || // A-Z
      (cp >= 0x0061u && cp <= 0x007Au);   // a-z
}

/**
 * eigen矩阵库计算argmax
 * @param half_ptr          数组首地址
 * @param n                 数组元素个数
 * @return                  返回最大值索引
 */
static int eigen_argmax_half(const half *half_ptr, std::size_t n)
{
  using Eigen::Tensor;
  using Eigen::TensorMap;

  TensorMap<Tensor<const half, 1>> tm(static_cast<const half *>(half_ptr), n);

  Eigen::Tensor<Eigen::Index, 0> t = tm.argmax();
  Eigen::Index idx = t(0);

  return static_cast<int>(idx);
}

#endif // __UTILS_H__