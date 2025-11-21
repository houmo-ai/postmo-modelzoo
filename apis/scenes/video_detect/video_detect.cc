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
#include "video_codec.h"

void CalculateSpeed(std::vector<std::shared_ptr<VideoCodec>>& codec_vec,
                    bool& stop_flag) {
  auto codec_num = codec_vec.size();
  LOG_INFO("==> Start to monitor encoding/decoding speed, codec num is {}.",
           codec_num);
  auto start = GET_TIME();
  std::vector<int32_t> decoded_hist_vec(codec_num, 0);
  std::vector<int32_t> encoded_hist_vec(codec_num, 0);
  std::vector<std::chrono::system_clock::time_point> previous_time(codec_num,
                                                                   start);
  while (!stop_flag) {
    for (int idx = 0; idx < codec_num; idx++) {
      auto codec = codec_vec[idx];
      auto encoded_cnt = codec->encoded_cnt;
      auto decoded_cnt = codec->decoded_cnt;
      auto current = GET_TIME();
      int diff_enc_cnt = encoded_cnt - encoded_hist_vec[idx];
      int diff_dec_cnt = decoded_cnt - decoded_hist_vec[idx];
      float diff_time =
        (GET_COST(previous_time[idx], current) / 1000.0 / 1000.0);
      encoded_hist_vec[idx] = encoded_cnt;
      decoded_hist_vec[idx] = decoded_cnt;
      previous_time[idx] = current;
      float enc_speed = 1.0 * diff_enc_cnt / diff_time;
      float dec_speed = 1.0 * diff_dec_cnt / diff_time;
      LOG_INFO(
        "Codec Stats: {} decoding speed: {} fps, encoding speed: {} fps.",
        reinterpret_cast<void*>(codec.get()), dec_speed, enc_speed);
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  }
  LOG_INFO("<== End to monitor encoding/decoding speed.");
}

PooledModule* LoadModelFromFile(ModulePool* module_pool, std::string model_path,
                                tcim::Module::WeightManager& wm,
                                std::vector<std::string>& dummy_tensor_names) {
  if (!fs::exists(model_path)) {
    LOG_ERROR("Model file {} doesn't exist.", model_path);
    return nullptr;
  }

  tcim::Module::Option option(wm);
  option.SetDummyTensors(dummy_tensor_names);
  auto pooled_md = module_pool->Load(model_path, option);
  if (pooled_md == nullptr) {
    LOG_ERROR("Failed to load model {}.", model_path);
    return nullptr;
  }
  LOG_INFO("Load model {} and pool it into {}.", model_path,
           reinterpret_cast<void*>(pooled_md));

  return pooled_md;
}

int main(int argc, char** argv) {
  LOG_WARNING("[MEM PERF] Main Start, current mem is {} MB",
              getCurrentMemoryUsage());

  int codec_thread_num = 2;
  if (argc > 1) {
    codec_thread_num = std::stoi(argv[1]);
    LOG_INFO("set codec thread number: {}", codec_thread_num);
  }

  std::string det_model_path = "./yolov5s.hmm";
  std::string cls_model_path = "./resnet50.hmm";
  std::string stream_path = "../../data/1080P_traffic_4s.h264";
  int det_thread_num = codec_thread_num;
  int cls_thread_num = det_thread_num;
  size_t frame_limit = 0;

  std::vector<std::thread> threads;
  std::vector<TaskQueue> detect_queues(codec_thread_num);
  std::vector<TaskQueue> classify_queues(codec_thread_num);
  std::vector<TaskQueue> encoding_queues(codec_thread_num);

  if (auto platform = std::getenv("HDPL_PLATFORM")) {
    if (!strcmp(platform, "ISIM")) {
      det_thread_num = 1;
      cls_thread_num = cls_thread_num == 0 ? 0 : 1;
      frame_limit = 2;
      LOG_WARNING("det_thread_num set to {} while HDPL_PLATFORM=ISIM",
                  det_thread_num);
      LOG_WARNING("cls_thread_num set to {} while HDPL_PLATFORM=ISIM",
                  cls_thread_num);
      LOG_WARNING("frame_limit set to {} while HDPL_PLATFORM=ISIM",
                  frame_limit);
    }
  }

  auto postfix = stream_path.substr(stream_path.size() - 5);
  std::string format;
  if (postfix == ".h264") {
    format = "H264";
  } else if (postfix == ".h265") {
    format = "H265";
  } else {
    LOG_ERROR("file format not supported: {}", stream_path);
    exit(-1);
  }

#ifndef DECODER_ONLY
  bool classify_task = cls_thread_num == 0 ? false : true;
  int module_max_num = 4;
  int stream_num = 4;

  // create module pool instance
  auto pool_ptr = ModulePool::Init(module_max_num, stream_num);
  // create infer threads
  int infer_thread_num = det_thread_num + cls_thread_num;
  Barrier barrier(infer_thread_num);
  std::vector<InferInfo> det_infer_infos(det_thread_num);
  std::vector<InferInfo> cls_infer_infos(cls_thread_num);
  auto wm = tcim::Module::WeightManager::CreateWeightManager(DEVICE_ID);
  // Load yolov5s model and create detection threads
  std::vector<std::string> yolov5s_names = {"images", "dyn_info"};
  for (int idx = 0; idx < det_thread_num; idx++) {
    InferInfo detect_info;
    PooledModule* yolov5s_md =
      LoadModelFromFile(pool_ptr, det_model_path, wm, yolov5s_names);
    if (yolov5s_md == nullptr) {
      LOG_ERROR("Failed to load yolov5s model: {}", det_model_path);
      exit(-1);
    }
    LOG_INFO("thread {} detection model loaded: {}.", idx, det_model_path);
    detect_info.module = yolov5s_md;
    detect_info.id = idx;
    det_infer_infos[idx] = detect_info;
    threads.emplace_back(std::thread(
      &detect, std::ref(det_infer_infos[idx]), std::ref(detect_queues[idx]),
      std::ref(classify_queues[idx]), std::ref(encoding_queues[idx]),
      classify_task, std::ref(barrier)));
  }

  // Load resnet50 model and create classify threads
  std::vector<std::string> resnet50_names = {"input.1"};
  for (int idx = 0; idx < cls_thread_num; idx++) {
    InferInfo classify_info;
    PooledModule* resnet50_md =
      LoadModelFromFile(pool_ptr, cls_model_path, wm, resnet50_names);
    if (resnet50_md == nullptr) {
      LOG_ERROR("Failed to load resnet50 model: {}", cls_model_path);
      exit(-1);
    }
    LOG_INFO("thread {} classify model loaded: {}.", (det_thread_num + idx),
             cls_model_path);
    classify_info.module = resnet50_md;
    classify_info.id = (det_thread_num + idx);
    cls_infer_infos[idx] = classify_info;
    threads.emplace_back(std::thread(&classify, std::ref(cls_infer_infos[idx]),
                                     std::ref(classify_queues[idx]),
                                     std::ref(barrier)));
  }
  barrier.wait();
#endif  // !DECODER_ONLY

  // yolov5s input shape (width, height)
  int32_t width = 1920;
  int32_t height = 1080;
  bool stop_flag = false;
  std::vector<std::shared_ptr<VideoCodec>> codec_vec;
  PushStreamInfo stream_info = {stream_path, frame_limit};
#ifdef RK_DECODER
  int barrier_multiple = 1;
#else
  int barrier_multiple = 2;
  std::vector<std::shared_ptr<VideoDecoder>> decoder_vec;
#endif
#ifdef ENC_TASK
  barrier_multiple += 2;
  std::vector<std::shared_ptr<VideoEncoder>> encoder_vec;
#endif
  Barrier barrier2((codec_thread_num * barrier_multiple));

  for (int codec_idx = 0; codec_idx < codec_thread_num; codec_idx++) {
    std::shared_ptr<VideoCodec> codec = std::make_shared<VideoCodec>();
    codec_vec.emplace_back(codec);
#ifdef RK_DECODER
    threads.emplace_back(
      std::thread(&VideoCodec::PushRKStream, codec, std::ref(stream_info),
                  std::ref(detect_queues[codec_idx]), std::ref(barrier2),
                  codec_thread_num, std::ref(stop_flag)));
#else
    std::shared_ptr<VideoDecoder> decoder(new VideoDecoder(format, "NV12M"));
#ifdef RESIZER
    decoder->SetModelInfo(width, height);
#endif  // RESIZER
    decoder_vec.emplace_back(decoder);
    threads.emplace_back(
      std::thread(&VideoCodec::PushStream, codec, decoder_vec[codec_idx],
                  std::ref(stream_info), std::ref(barrier2)));
    threads.emplace_back(
      std::thread(&VideoCodec::GetFrame, codec, decoder_vec[codec_idx],
                  std::ref(detect_queues[codec_idx]), std::ref(barrier2),
                  codec_thread_num, std::ref(stop_flag)));
#endif  // RK_DECODER

#ifndef DECODER_ONLY
#ifdef ENC_TASK
    std::string output_path =
      "encoded_results_video_" + std::to_string(codec_idx) + ".h264";
    std::shared_ptr<VideoEncoder> encoder(
      new VideoEncoder("NV12M", format, width, height));
    encoder_vec.emplace_back(encoder);
    threads.emplace_back(
      std::thread(&VideoCodec::PushEncodeStream, codec, encoder_vec[codec_idx],
                  std::ref(encoding_queues[codec_idx]), std::ref(barrier2)));
    threads.emplace_back(std::thread(&VideoCodec::GetEncodeStream, codec,
                                     encoder_vec[codec_idx], width, height,
                                     output_path, codec_thread_num,
                                     std::ref(stop_flag), std::ref(barrier2)));
#endif  // ENC_TASK
#endif  // !DECODER_ONLY
  }

  // create a thread to calculate encoding/decoding speed
  threads.emplace_back(
    std::thread(&CalculateSpeed, std::ref(codec_vec), std::ref(stop_flag)));
  barrier2.wait();

  for (auto& t : threads) {
    t.join();
  }

#ifndef DECODER_ONLY
  for (auto& infer_info : det_infer_infos) {
    delete infer_info.module;
    infer_info.module = nullptr;
  }
  det_infer_infos.clear();
  for (auto& infer_info : cls_infer_infos) {
    delete infer_info.module;
    infer_info.module = nullptr;
  }
  cls_infer_infos.clear();
  delete pool_ptr;
#endif

  return 0;
}
