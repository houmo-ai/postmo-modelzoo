#include <getopt.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "detect_frames.h"
#include "jpeg_codec.h"

int main(int argc, char **argv) {
  std::string det_model_path = "yolov5s.hmm";
  std::string cls_model_path = "resnet50.hmm";
  int det_thread_num = 1;
  int cls_thread_num = 2;

  std::vector<std::thread> threads;
  TaskQueue q_enc;
  TaskQueue q_det;
  TaskQueue q_cls;

  // create infer threads
  int infer_thread_num = det_thread_num + cls_thread_num;
  Barrier barrier(infer_thread_num);
  std::vector<InferInfo> det_infer_infos(det_thread_num);
  std::vector<InferInfo> cls_infer_infos(cls_thread_num);

  auto wm = tcim::Module::WeightManager::CreateWeightManager(DEVICE_ID);
  for (int i = 0; i < det_thread_num; i++) {
    InferInfo infer_info;
    if (infer_info.module.Load(det_model_path, wm)) {
      LOG_ERROR("load model fail: {}", det_model_path);
      exit(-1);
    }
    LOG_INFO("thread {} model loaded: {}.", i, det_model_path);
    infer_info.id = i;
    det_infer_infos[i] = infer_info;
    threads.emplace_back(std::thread(&detect, std::ref(det_infer_infos[i]),
                                     std::ref(q_det), std::ref(q_cls),
                                     std::ref(q_enc), std::ref(barrier)));
  }
  for (int i = 0; i < cls_thread_num; i++) {
    InferInfo infer_info;
    if (infer_info.module.Load(cls_model_path, wm)) {
      LOG_ERROR("load model fail: {}", cls_model_path);
      exit(-1);
    }
    LOG_INFO("thread {} model loaded: {}.", i, cls_model_path);
    infer_info.id = i;
    cls_infer_infos[i] = infer_info;
    threads.emplace_back(std::thread(&classify, std::ref(cls_infer_infos[i]),
                                     std::ref(q_cls), std::ref(barrier)));
  }
  wm.~WeightManager();
  barrier.wait();

  // JPEG codec params
  int encode_num = 20;
  int width = 640;
  int height = 426;
  std::string data_path = "../../data/000000000139.jpg";

  const int y_size = width * height;
  const int uv_size = y_size / 2;
  const int yuv_total_size = y_size + uv_size;

  cv::Mat img_rgb;
  cv::Mat img_yuv;
  img_rgb = cv::imread(data_path);
  ImageProc::BgrToRgb((int8_t *)(img_rgb.data), img_rgb.rows, img_rgb.cols);
  cv::cvtColor(img_rgb, img_yuv, cv::COLOR_RGB2YUV_I420);
  int size = width * height * 3;
  char *yuv_buffer = (char *)(malloc(yuv_total_size));
  ImageProc::I420To420sp((uint8_t *)yuv_buffer, (uint8_t *)img_yuv.data, size);

  JpegCodec codec;
  Barrier barrier2(2);
  JpegEncoder encoder("NV12M", "JPEG", width, height);
  JpegDecoder decoder("JPEG", "NV12M");
  decoder.SetModelInfo(1920, 1080);  // yolov5s input shape (width, height)

  // create jpeg encoder thread
  threads.emplace_back(std::thread(&JpegCodec::EncodeImage, &codec,
                                   std::ref(encoder), yuv_buffer, encode_num,
                                   std::ref(q_enc), std::ref(barrier2)));
  // create jpeg decoder thread
  threads.emplace_back(std::thread(
      &JpegCodec::DecodeImage, &codec, std::ref(decoder), width, height,
      std::ref(q_enc), std::ref(q_det), std::ref(barrier2)));
  barrier2.wait();

  for (auto &t : threads) {
    t.join();
  }

  return 0;
}
