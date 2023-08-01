#include "hdpl_yolo_layer.h"

#include <math.h>
#include <stdio.h>

#include <iostream>

#include "box.h"

//
// Code From
// https://github.com/ChenYingpeng/caffe-yolov3/blob/master/src/yolo_layer.cpp
//

// yolov3
int biases[18] = {10, 13, 16,  30,  33, 23,  30,  61,  62,
                  45, 59, 119, 116, 90, 156, 198, 373, 326};

// yolov3-tiny
float biases_tiny[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};

int max_index(float *a, int n) {
  if (n <= 0) return -1;
  int i, max_i = 0;
  float max = a[0];
  for (i = 1; i < n; ++i) {
    if (a[i] > max) {
      max = a[i];
      max_i = i;
    }
  }
  return max_i;
}

layer make_yolo_layer(int batch, int w, int h, int net_w, int net_h, int n,
                      int total, int classes) {
  layer l = {0};
  l.n = n;
  l.total = total;
  l.batch = batch;
  l.h = h;
  l.w = w;
  l.c = n * (classes + 4 + 1);
  l.out_w = l.w;
  l.out_h = l.h;
  l.out_c = l.c;
  l.classes = classes;
  l.inputs = l.w * l.h * l.c;

  l.biases = reinterpret_cast<int *>(calloc(total * 2, sizeof(int)));

  l.mask = reinterpret_cast<int *>(calloc(n, sizeof(int)));
  if (9 == total) {
    for (int i = 0; i < total * 2; ++i) {
      l.biases[i] = biases[i];
    }
    if (l.w == net_w / 32) {
      int j = 6;
      for (int i = 0; i < l.n; ++i) l.mask[i] = j++;
    }
    if (l.w == net_w / 16) {
      int j = 3;
      for (int i = 0; i < l.n; ++i) l.mask[i] = j++;
    }
    if (l.w == net_w / 8) {
      int j = 0;
      for (int i = 0; i < l.n; ++i) l.mask[i] = j++;
    }
  }

  if (6 == total) {
    for (int i = 0; i < total * 2; ++i) {
      l.biases[i] = biases_tiny[i];
    }
    if (l.w == net_w / 32) {
      int j = 3;
      for (int i = 0; i < l.n; ++i) l.mask[i] = j++;
    }
    if (l.w == net_w / 16) {
      int j = 0;
      for (int i = 0; i < l.n; ++i) l.mask[i] = j++;
    }
  }
  l.outputs = l.inputs;
  return l;
}

void free_yolo_layer(layer l) {
  if (NULL != l.biases) {
    free(l.biases);
    l.biases = NULL;
  }

  if (NULL != l.mask) {
    free(l.mask);
    l.mask = NULL;
  }
}

static int entry_index(layer l, int batch, int location, int entry) {
  int n = location / (l.w * l.h);
  int loc = location % (l.w * l.h);
  return batch * l.outputs + n * l.w * l.h * (4 + l.classes + 1) +
         entry * l.w * l.h + loc;
}

int yolo_num_detections(layer l, float thresh) {
  int i, n, b;
  int count = 0;
  for (b = 0; b < l.batch; ++b) {
    for (i = 0; i < l.w * l.h; ++i) {
      for (n = 0; n < l.n; ++n) {
        int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
        if (l.output[obj_index] > thresh) ++count;
      }
    }
  }
  return count;
}

int num_detections(std::vector<layer> layers_params, float thresh) {
  int i;
  int s = 0;
  for (i = 0; i < layers_params.size(); ++i) {
    layer l = layers_params[i];
    s += yolo_num_detections(l, thresh);
  }
  return s;
}

detection *make_network_boxes(const std::vector<layer> &layers_params,
                              float thresh, int *num) {
  const layer l = layers_params[0];
  int i;
  int nboxes = num_detections(layers_params, thresh);
  if (num) *num = nboxes;
  detection *dets =
      reinterpret_cast<detection *>(calloc(nboxes, sizeof(detection)));
  for (i = 0; i < nboxes; ++i) {
    dets[i].prob = reinterpret_cast<float *>(calloc(l.classes, sizeof(float)));
  }
  return dets;
}

void correct_yolo_boxes(detection *dets, int n, int w, int h, int netw,
                        int neth, int relative) {
  int i;
  int new_w = 0;
  int new_h = 0;
  if ((static_cast<float>(netw) / w) < (static_cast<float>(neth) / h)) {
    new_w = netw;
    new_h = (h * netw) / w;
  } else {
    new_h = neth;
    new_w = (w * neth) / h;
  }
  for (i = 0; i < n; ++i) {
    box b = dets[i].bbox;
    b.x = (b.x - (netw - new_w) / 2. / netw) /
          (static_cast<float>(new_w) / static_cast<float>(netw));
    b.y = (b.y - (neth - new_h) / 2. / neth) /
          (static_cast<float>(new_h) / static_cast<float>(neth));
    b.w *= static_cast<float>(netw) / new_w;
    b.h *= static_cast<float>(neth) / new_h;
    if (!relative) {
      b.x *= w;
      b.w *= w;
      b.y *= h;
      b.h *= h;
    }
    dets[i].bbox = b;
  }
}

box get_yolo_box(const float *x, int *biases, int n, int index, int i, int j,
                 int lw, int lh, int w, int h, int stride) {
  box b;
  b.x = (i + x[index + 0 * stride]) / lw;
  b.y = (j + x[index + 1 * stride]) / lh;
  b.w = exp(x[index + 2 * stride]) * biases[2 * n] / w;
  b.h = exp(x[index + 3 * stride]) * biases[2 * n + 1] / h;
  return b;
}

int get_yolo_detections(layer l, int w, int h, int netw, int neth,
                        float conf_thresh, int *map, int relative,
                        detection *dets) {
  int i, j, n, b;
  const float *predictions = l.output;
  int count = 0;
  for (b = 0; b < l.batch; ++b) {
    for (i = 0; i < l.w * l.h; ++i) {
      int row = i / l.w;
      int col = i % l.w;
      for (n = 0; n < l.n; ++n) {
        int obj_index = entry_index(l, b, n * l.w * l.h + i, 4);
        float objectness = predictions[obj_index];
        if (objectness <= conf_thresh) continue;
        int box_index = entry_index(l, b, n * l.w * l.h + i, 0);
        dets[count].bbox =
            get_yolo_box(predictions, l.biases, l.mask[n], box_index, col, row,
                         l.w, l.h, netw, neth, l.w * l.h);
        dets[count].objectness = objectness;
        dets[count].classes = l.classes;
        dets[count].obj_index = obj_index;
        for (j = 0; j < l.classes; ++j) {
          int class_index = entry_index(l, b, n * l.w * l.h + i, 4 + 1 + j);
          // printf("class_index is %d\r\n",class_index);
          float prob = objectness * predictions[class_index];
          // printf("++count=%d j=%d\r\n",count,j);
          dets[count].prob[j] = (prob > conf_thresh) ? prob : 0;
          // printf("++after count\r\n");
        }
        ++count;
      }
    }
  }
  correct_yolo_boxes(dets, count, w, h, netw, neth, relative);
  return count;
}

void fill_network_boxes(const std::vector<layer> &layers_params, int img_w,
                        int img_h, int net_w, int net_h, float conf_thresh,
                        int *map, int relative, detection *dets) {
  int j;
  for (j = 0; j < layers_params.size(); ++j) {
    layer l = layers_params[j];
    int count = get_yolo_detections(l, img_w, img_h, net_w, net_h, conf_thresh,
                                    map, relative, dets);
    dets += count;
  }
}

detection *get_network_boxes(const std::vector<layer> &layers_params, int img_w,
                             int img_h, int net_w, int net_h, float conf_thresh,
                             int *map, int relative, int *num) {
  // make network boxes
  detection *dets = make_network_boxes(layers_params, conf_thresh, num);

  // fill network boxes
  fill_network_boxes(layers_params, img_w, img_h, net_w, net_h, conf_thresh,
                     map, relative, dets);
  return dets;
}

// get detection result
detection *get_detections(std::vector<Blob<float> *> blobs, int img_w,
                          int img_h, int net_w, int net_h, float nms_thresh,
                          float conf_thresh, int classes, int *nboxes) {
  std::vector<layer> layers_params;
  layers_params.clear();
  for (int i = 0; i < blobs.size(); ++i) {
    layer l_params;
    l_params = make_yolo_layer(blobs[i]->num(), blobs[i]->width(),
                               blobs[i]->height(), net_w, net_h, num_bboxes,
                               blobs.size() * dev_num_anchors, classes);
    l_params.output = blobs[i]->data_;
    layers_params.push_back(l_params);
  }

  // get network boxes
  detection *dets = get_network_boxes(layers_params, img_w, img_h, net_w, net_h,
                                      conf_thresh, 0, relative, nboxes);

  // release layer memory
  for (int index = 0; index < layers_params.size(); ++index) {
    free_yolo_layer(layers_params[index]);
  }

  // do nms
  if (nms_thresh) do_nms_sort(dets, (*nboxes), classes, nms_thresh);

  return dets;
}

// release detection memory
void free_detections(detection *dets, int nboxes) {
  int i;
  for (i = 0; i < nboxes; ++i) {
    free(dets[i].prob);
  }
  free(dets);
}

std::ostream &operator<<(std::ostream &os, const detection &value) {
  os << "\"detection\": {\"classes=\"" << value.classes
     << " \"index\"=" << value.obj_index << " \"prob\"=[";
  // vim lfor (size_t idx=0;idx<value.classes;++idx) {
  for (size_t idx = 0; idx < 80; ++idx) {
    if (value.prob[idx] > 0) {
      os << idx << ": " << value.prob[idx] << ", ";
    }
  }
  os << "] \"objectness\"=" << value.objectness
     << " \"sort_class\"=" << value.sort_class << " \"x\"=" << value.bbox.x
     << " \"y\"=" << value.bbox.y << " \"w\"=" << value.bbox.w
     << " \"h\"=" << value.bbox.h << "}";
  return os;
}

std::ostream &operator<<(std::ostream &os, const layer &value) {
  os << "\"layer\": {\"batch=\"" << value.batch << " \"total\"=" << value.total
     << " \"n\"=" << value.n << " \"c\"=" << value.c << " \"h\"=" << value.h
     << " \"w\"=" << value.w << " \"out_n\"=" << value.out_n
     << " \"out_c\"=" << value.out_c << " \"out_h\"=" << value.out_h
     << " \"out_w\"=" << value.out_w << " \"classes\"=" << value.classes
     << " \"inputs\"=" << value.inputs << " \"outputs\"=" << value.outputs
     << "}";
  return os;
}
