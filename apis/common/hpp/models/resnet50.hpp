#include <vector>
#include <algorithm>


typedef struct ClassResult {
  int index;
  float conf;

  ClassResult() {}
  ClassResult(int index, float conf) : index(index), conf(conf) {}
} ClassResult;


class Resnet50 {
 public:

  std::vector<ClassResult> postprocess(float* output, int len) {
    std::vector<ClassResult> sort_pairs;
    for (int i = 0; i < len; ++i) {
      sort_pairs.emplace_back(i, output[i]);
    }
    std::sort(sort_pairs.begin(), sort_pairs.end(),
              [](const ClassResult& a, ClassResult& b) {
                return a.conf > b.conf;
              });
    return sort_pairs;
  }

  const int input_sizes_[2] = {224, 224}; // wh
};