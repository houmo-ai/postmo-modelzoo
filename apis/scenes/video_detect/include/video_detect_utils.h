#ifndef EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_VIDEO_DETECT_UTILS_H_
#define EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_VIDEO_DETECT_UTILS_H_

#include <condition_variable>
#include <mutex>
#include <queue>
#include <string>

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <opencv2/opencv.hpp>

#include "logging.h"
#include "models/resnet50.hpp"
#include "models/yolov5s.hpp"
#include "tcim/tcim_imageops.h"
#include "tcim/tcim_runtime.h"
#include "threads.hpp"
#include "utils.hpp"

#define DECODER_QUEUE_SIZE 20
#define INFERENCE_QUEUE_SIZE 10

// the maximum size supported by resizer
#define RESIZER_MAX_WIDTH 3840
#define RESIZER_MAX_HEIGHT 2160

#define VIDEO_DECODER_ERR -1
#define VIDEO_DECODER_OK 0
#define VIDEO_DECODER_EOS 1

#define DEVICE_ID 0

typedef enum {
  CPU = 0,
  HDPL = 1,
} FrameDevice;

typedef struct {
  void *data = nullptr;
  size_t len = 0;
} EncodedData;

typedef struct {
  void *data = nullptr;
  size_t len = 0;
} DecodeData;

typedef struct {
  tcim::ImageOps::Resizer *resizer = nullptr;
  int64_t max_height = 0;
  int64_t max_width = 0;
  int64_t md_height = 0;
  int64_t md_width = 0;
} ImgResizer;

typedef struct ObjInfo {
  DetectResult det;
  ClassResult cls;
} ObjInfo;

typedef struct {
  tcim::Buffer image;
  std::vector<ObjInfo> objs;
  uint64_t req_id;
  bool is_end = false;
#ifdef RK_DECODER
  // only used in rk decoder
  std::shared_ptr<void> buffer;
  size_t buffer_length;
#endif
} TaskInfo;

typedef struct {
  std::queue<TaskInfo> queue;
  std::mutex mutex;
  std::condition_variable cond;
} TaskQueue;

typedef struct {
  std::string stream_path;
  uint64_t frame_limit;
} PushStreamInfo;

inline int SaveImgs(int height, int width, void *data_ptr,
                    std::string folder_name, std::string file_name) {
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, data_ptr);
  cv::Mat rgb;
  cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_BGR2RGB);

  fs::path file_path(file_name + ".jpg");
  fs::path result_path(folder_name);
  if (!fs::exists(result_path)) {
    fs::create_directory(folder_name);
  }
  fs::path result_file = result_path / file_path.filename();
  cv::imwrite(result_file.string().c_str(), bgr);

  return VIDEO_DECODER_OK;
}

#endif  // EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_VIDEO_DETECT_UTILS_H_