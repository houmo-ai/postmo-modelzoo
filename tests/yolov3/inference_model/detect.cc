#include "detect.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <tuple>

#include "annotation.h"
#include "hdpl_yolo_layer.h"

struct OutInfo {
  std::vector<float> tvm_fmt_out;
  int n;
  int c;
  int h;
  int w;
};

OutInfo get_tvm_fmt_out(const float *, int, int, int, int);

bool CaltulateDetection(std::vector<DetectInfo> *detections,
                        const ImageInfo &image_info,
                        const std::vector<OutInfo> &infos);

bool yolo_detect(std::vector<DetectInfo> *detections, const float *big_output,
                 const float *mid_output, const float *small_output,
                 const ImageInfo &image_info) {
  std::vector<OutInfo> infos;
  infos.push_back(get_tvm_fmt_out(small_output, 1, 255, 13, 13));
  infos.push_back(get_tvm_fmt_out(mid_output, 1, 255, 26, 26));
  infos.push_back(get_tvm_fmt_out(big_output, 1, 255, 52, 52));
  return CaltulateDetection(detections, image_info, infos);
}

//
// Code From
// https://github.com/ChenYingpeng/caffe-yolov3/blob/master/src/activation_kernels.cu#L27
//
float sigmoid(float x) { return 1.0 / (1 + exp(-x)); }

OutInfo get_tvm_fmt_out(const float *data, int n, int c, int h, int w) {
  OutInfo out_info;
  auto size = n * h * w * c;
  n = 3;
  c = c / n;
  out_info.tvm_fmt_out.resize(size, 0);
  for (int ni = 0; ni < n; ni++) {
    for (int ci = 0; ci < c; ci++) {
      for (int hi = 0; hi < h; hi++) {
        for (int wi = 0; wi < w; wi++) {
          if (ci < 2 || (ci >= 4 && ci < 85)) {
            out_info.tvm_fmt_out[ni * c * h * w + ci * h * w + hi * w + wi] =
                sigmoid(data[ni * c * h * w + ci * h * w + hi * w + wi]);
          } else {
            out_info.tvm_fmt_out[ni * c * h * w + ci * h * w + hi * w + wi] =
                data[ni * c * h * w + ci * h * w + hi * w + wi];
          }
        }
      }
    }
  }
  out_info.n = n;
  out_info.c = c;
  out_info.h = h;
  out_info.w = w;
  return out_info;
}

/**
 * @brief
 *
 * @param detections
 * @param image_info
 * @param infos
 * @return true
 * @return false
 */
bool CaltulateDetection(std::vector<DetectInfo> *detections,
                        const ImageInfo &image_info,
                        const std::vector<OutInfo> &infos) {
  detection *dets = NULL;
  float nms_thresh = 0.5;
  int m_classes = 80;  // coco classes
  int nboxes = 0;
  float conf_thresh = 0.001;
  std::vector<Blob<float> *> m_blobs_;
  for (size_t n = 0; n < infos.size(); n++) {
    m_blobs_.push_back(new Blob<float>);
  }

  for (size_t i = 0; i < m_blobs_.size(); ++i) {
    m_blobs_[i]->shape_ = {infos[i].n, infos[i].c, infos[i].h, infos[i].w};
    m_blobs_[i]->data_ = infos[i].tvm_fmt_out.data();
  }
  dets = get_detections(m_blobs_, image_info.width, image_info.height, 416, 416,
                        nms_thresh, conf_thresh, m_classes, &nboxes);
  for (auto m : m_blobs_) delete m;

  // deal with results
  for (int i = 0; i < nboxes; ++i) {
    box b = dets[i].bbox;
    int const obj_id = max_index(dets[i].prob, m_classes);
    float const prob = dets[i].prob[obj_id];

    if (prob > conf_thresh) {
      // x, y from get_detections are the center point of the object
      // convert the center point to the left top point
      float obj_x = std::max(0., (b.x - b.w / 2.) * image_info.width);
      float obj_y = std::max(0., (b.y - b.h / 2.) * image_info.height);
      float obj_width = b.w * image_info.width;
      float obj_height = b.h * image_info.height;
      // The detection format is {obj_id, x, y, w, h, confidence, class_id}
      // The format is required by COCO
      DetectInfo detect_info = {static_cast<float>(image_info.id),
                                obj_x,
                                obj_y,
                                obj_width,
                                obj_height,
                                prob,
                                static_cast<float>(obj_id + 1)};
      LOG(INFO) << " id = " << detect_info[0] << " class = " << detect_info[6]
                << " prob = " << detect_info[5] << " x = " << detect_info[1]
                << " y = " << detect_info[2] << " w = " << detect_info[3]
                << " h = " << detect_info[4];
      detections->push_back(detect_info);
    }
  }
  free_detections(dets, nboxes);
  return true;
}
