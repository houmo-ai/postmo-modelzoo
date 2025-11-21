#ifndef EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_DETECT_FRAMES_H_
#define EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_DETECT_FRAMES_H_

#include <stdio.h>
#include <unistd.h>

#include <condition_variable>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "datasets/imagenet.hpp"
#include "imageproc.hpp"
#include "infer_module.hpp"
#include "video_detect_utils.h"

typedef struct {
  PooledModule *module;
  int id = 0;
} InferInfo;

int GetInputInfoMap(PooledModule *module,
                    std::map<std::string, tcim::TensorInfo> &input_info_map);
int GetOutputInfoMap(PooledModule *module,
                     std::map<std::string, tcim::TensorInfo> &output_info_map);

void detect(InferInfo &infer_info, TaskQueue &qin, TaskQueue &qout,
            TaskQueue &qout_enc, bool classify_task, Barrier &barrier);
void classify(InferInfo &infer_info, TaskQueue &qin, Barrier &barrier);

#endif  // EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_DETECT_FRAMES_H_