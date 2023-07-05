#ifndef __YOLO_LAYER_H_
#define __YOLO_LAYER_H_
#include <iostream>
#include <string>
#include <vector>

const float hier_thresh = 0.5;
const float nms_thresh = 0.45;
const int num_bboxes = 3;
const int relative = 1;

const int dev_num_anchors = 3;

template <typename Dtype>
class Blob {
 public:
  /// @brief Deprecated legacy shape accessor num: use shape(0) instead.
  // inline int num() const { return shape_[0]; }
  inline int num() const { return 1; }
  /// @brief Deprecated legacy shape accessor channels: use shape(1) instead.
  inline int channels() const { return shape_[1]; }
  /// @brief Deprecated legacy shape accessor height: use shape(2) instead.
  inline int height() const { return shape_[2]; }
  /// @brief Deprecated legacy shape accessor width: use shape(3) instead.
  inline int width() const { return shape_[3]; }
  std::vector<int> shape_;

  const Dtype *data_;
  int count_;
  int capacity_;
  bool use_cuda_host_malloc_;
};  // class Blob

struct bbox_t {
  float x;
  float y;
  float w;
  float h;     // (x,y) - top-left corner, (w,h) - width & height of bounded box
  float prob;  // confidence - probability that the object was found correctly
  unsigned int obj_id;  // class of object - from range [0,classes - 1]
};

typedef struct {
  float x, y, w, h;
} box;

typedef struct {
  box bbox;
  int classes;
  float *prob;
  float objectness;
  int sort_class;
  int obj_index;
} detection;

std::ostream &operator<<(std::ostream &os, const detection &value);

typedef struct layer {
  int batch;
  int total;
  int n, c, h, w;
  int out_n, out_c, out_h, out_w;
  int classes;
  int inputs, outputs;
  int *mask;
  int *biases;
  const float *output;
} layer;

std::ostream &operator<<(std::ostream &os, const layer &value);

layer make_yolo_layer(int batch, int w, int h, int n, int total, int classes);

void free_yolo_layer(layer l);

void forward_yolo_layer_gpu(const float *input, layer l, float *output);

detection *get_detections(std::vector<Blob<float> *> blobs, int img_w,
                          int img_h, int net_w, int net_h, float thresh,
                          int classes, int *nboxes);

void free_detections(detection *dets, int nboxes);

int max_index(float *a, int n);

#endif
