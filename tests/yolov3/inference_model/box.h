#ifndef CV_YOLOV3_INFERENCE_MODEL_BOX_H_
#define CV_YOLOV3_INFERENCE_MODEL_BOX_H_

#include "hdpl_yolo_layer.h"

void do_nms_sort(detection *dets, int total, int classes, float thresh);

#endif  // CV_YOLOV3_INFERENCE_MODEL_BOX_H_
