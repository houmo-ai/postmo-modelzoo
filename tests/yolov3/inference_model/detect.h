#ifndef __YOLO_H__
#define __YOLO_H__

#include <vector>

#include "annotation.h"

typedef std::vector<float> DetectInfo;

bool yolo_detect(std::vector<DetectInfo> *detections, const float *big_output,
                 const float *mid_output, const float *small_output,
                 const ImageInfo &image_info);

#endif  // __YOLO_H__
