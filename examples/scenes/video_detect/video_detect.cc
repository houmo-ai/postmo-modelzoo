#include <getopt.h>
#include <stdio.h>
#include <unistd.h>
#include <iomanip>
#include <chrono>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>
#include <queue>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <linux/videodev2.h>

#if (__GNUC__ < 8 && !defined(_MSC_VER))
#include <experimental/filesystem>
namespace fs = std::experimental::filesystem;
#else
#include <filesystem>
namespace fs = std::filesystem;
#endif

#include <opencv2/opencv.hpp>
extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

#include "models/yolov5s.hpp"
#include "models/resnet50.hpp"
#include "datasets/imagenet.hpp"
#include "imageproc.hpp"
#include "threads.hpp"
#include "utils.hpp"
#include "log.hpp"
#include "infer_module.hpp"
#include "video_decoder.h"


typedef struct ObjInfo {
  DetectResult det;
  ClassResult cls;
} ObjInfo;

typedef struct {
  tcim::Buffer image;
  std::vector<ObjInfo> objs;
  uint64_t req_id;
  bool is_end = false;
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

typedef struct {
  InferModule module;
  int id = 0;
} InferInfo;

int SaveImgs(int height, int width, void* data_ptr, std::string file_name)
{
  cv::Mat nv12(height * 3 / 2, width, CV_8UC1, data_ptr);
  cv::Mat rgb;
  cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
  cv::Mat bgr;
  cv::cvtColor(rgb, bgr, cv::COLOR_BGR2RGB);

  fs::path file_path(file_name + ".jpg");
  fs::path result_path("debug_results");
  if (!fs::exists(result_path)) {
    fs::create_directory("debug_results");
  }
  fs::path result_file = result_path / file_path.filename();
  cv::imwrite(result_file.string().c_str(), bgr);

  return VIDEO_DECODER_OK;
}

void PushStream(VideoDecoder& decoder, PushStreamInfo& stream_info, Barrier& barrier)
{
  AVFormatContext *format_ctx = NULL;
  AVPacket *packet = NULL;
  int index_slice = 0;
  int ret = 0;
  uint32_t push_slices_cnt = 0;
  uint32_t rcv_I_slice_flag = 0;
  /* frame_interval and last_push_stamp unit is microseconds */
  // unsigned long frame_interval = 1000000 / input_stream_fps;
  unsigned long last_push_stamp = 0;
  unsigned long current_time = 0;
  int count = 0;

  /*
   * there maybe many streams in a media file (video stream or other audio streams).
   * check out check video stream
   */
  int video_stream_index = -1;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
  av_register_all();
#endif
  avformat_network_init();

  // av_log_set_level(AV_LOG_DEBUG);

  /* open stream */
  ret = avformat_open_input(&format_ctx, stream_info.stream_path.c_str(), NULL, NULL);
  if (ret < 0) {
    char error_buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, error_buf, sizeof(error_buf));
    LOG_ERROR << "open stream " << stream_info.stream_path << " failed: " << error_buf;
    return;
  }
  LOG_INFO << "open stream " << stream_info.stream_path << " completed.";

  ret = avformat_find_stream_info(format_ctx, NULL);
  if (ret < 0) {
    char error_buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, error_buf, sizeof(error_buf));
    LOG_ERROR << "find stream info failed: " << error_buf;
    return;
  }

  for (int i = 0; i < format_ctx->nb_streams; i++) {
    if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index = i;
      break;
    }
  }

  if (video_stream_index == -1) {
    LOG_ERROR << "can't find video stream";
    avformat_close_input(&format_ctx);
    return;
  }

  AVCodecParameters *codecParameters = format_ctx->streams[video_stream_index]->codecpar;
  LOG_INFO << "frame size is " << codecParameters->width << " x " <<  codecParameters->height;
  auto& width = decoder.GetWidth();
  width = codecParameters->width;
  auto& height = decoder.GetHeight();
  height = codecParameters->height;

#ifdef RESIZER
  if (width > RESIZER_MAX_WIDTH || height > RESIZER_MAX_HEIGHT) {
    LOG_ERROR << "the decoded outputs exceeds the upper limit supported by resizer.";
    avformat_close_input(&format_ctx);
    return;
  }
  // create a resizer to process decoded outputs
  auto option = tcim::ImageOps::Resizer::Option(tcim::DataFmt::YUV420SP, 0);
  auto md_width = decoder.GetResizer().md_width;
  auto md_height = decoder.GetResizer().md_width;
  int64_t max_width = width < md_width ? md_width : width;
  int64_t max_height = height < md_height ? md_height : height;
  option.SetMaxSize(max_width, max_height);
  if (decoder.SetResizer(option) != VIDEO_DECODER_OK) {
      LOG_ERROR << "create resizer fail.";
      avformat_close_input(&format_ctx);
      return;
  }
#endif

  barrier.barrier();

  packet = av_packet_alloc();
  if (packet == NULL) {
    LOG_ERROR << "packet failed";
    avformat_close_input(&format_ctx);
    return;
  }

  while (1) {
    if (count >= stream_info.frame_limit && stream_info.frame_limit != 0) {
      LOG_WARN << "===> push stream frame limit reached: " << stream_info.frame_limit;
      /* push last flag */
      ret = decoder.PushData(nullptr, 0, 1);
      if (ret != 0) {
        LOG_ERROR << "push last stream failed, ret: " << ret;
        ret = -1;
      }
      break;
    }
    ret = av_read_frame(format_ctx, packet);
    if (ret < 0) {
      if (packet->size == 0) {
        LOG_INFO << "===> push stream thread EOS received";
        /* push last flag */
        ret = decoder.PushData(packet->data, packet->size, 1);
        if (ret != 0) {
          LOG_ERROR << "push last stream failed, ret: " << ret;
          ret = -1;
        }
      } else {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        LOG_ERROR << "av_read_frame fail: " << error_buf;
        ret = -1;
      }
      /* exit loop */
      break;
    }

    /* wait I slice, when decoder start work. */
    if (packet->flags & AV_PKT_FLAG_KEY)
      rcv_I_slice_flag = 1;

    if (rcv_I_slice_flag != 1) {
      LOG_DEBUG << "skip non-key frame";
      av_packet_unref(packet);
      continue;
    }

    if (packet->stream_index == video_stream_index) {

      /* find which buf is available. if find, so push stream */
      push_slices_cnt++;

      ret = decoder.PushData(packet->data, packet->size, 0);
      if (ret != 0) {
        LOG_ERROR << "push stream failed: " << ret;
        ret = -1;
        break;
      }

      count++;
    }
    av_packet_unref(packet);
    // std::this_thread::sleep_for(std::chrono::milliseconds(200));
  }

  av_packet_free(&packet);
  avformat_close_input(&format_ctx);

  LOG_INFO << "<=== PushStream thread exit. " << count << " frames received.";
}

void GetFrame(VideoDecoder& decoder, TaskQueue& qout, Barrier& barrier) {
  int fid = 0;
  int ret = 0;
  int count = 0;

  FrameDevice device = HDPL;

  barrier.barrier();
  int height = decoder.GetHeight();
  int width = decoder.GetWidth();
  LOG_INFO << "===> decoder heigth:" << height << ", width:" << width;

  std::vector<DecodeData> frm_data(2);

  while (1) {
    ret = decoder.PullData(frm_data, device);
    if (ret == VIDEO_DECODER_EOS) {
      TaskInfo task_info;
      task_info.is_end = true;
      std::unique_lock<std::mutex> lock(qout.mutex);
      qout.queue.push(task_info);
      lock.unlock();
      LOG_INFO << "===> get frame thread EOS received.";
      break;
    }
    if (ret != VIDEO_DECODER_OK) {
      LOG_ERROR << "decoder pull data fail: " << ret;
      continue;
    }

    tcim::Buffer y_buf;
    tcim::Buffer uv_buf;
    tcim::Tensor yuv_tensor;
    tcim::Tensor y_tensor;
    tcim::Tensor uv_tensor;
    auto yuv_info = tcim::TensorInfo::CreateYUVInfo(width, height, tcim::YUV420SP);
    LOG_INFO << "-->> decoded yuv info:" << yuv_info;
    if (device == CPU) {
      yuv_tensor = tcim::Tensor::CreateHostTensor(yuv_info);
      y_buf = tcim::Buffer::CreateHostBuffer(frm_data[0].len, frm_data[0].data);
      uv_buf = tcim::Buffer::CreateHostBuffer(frm_data[1].len, frm_data[1].data);
    } else {
      yuv_tensor = tcim::Tensor::CreateDeviceTensor(yuv_info);
      y_buf = tcim::Buffer::CreateDeviceBuffer(frm_data[0].data, frm_data[0].len, 0, "");
      uv_buf = tcim::Buffer::CreateDeviceBuffer(frm_data[1].data, frm_data[1].len, 0, "");
    }
    yuv_tensor.SplitYUV(y_tensor, uv_tensor);
    y_buf.CopyTo(y_tensor.Buffer());
    uv_buf.CopyTo(uv_tensor.Buffer());

#ifdef RESIZER
    auto img_resizer = decoder.GetResizer();
    auto md_heigth = img_resizer.md_height;
    auto md_width = img_resizer.md_width;
    tcim::Tensor resized_yuv_tensor;
    auto resized_yuv_info = tcim::TensorInfo::CreateYUVInfo(
      md_width, md_heigth, tcim::YUV420SP);
    LOG_INFO << "-->> resized yuv info:" << resized_yuv_info;
    if (device == CPU) {
      resized_yuv_tensor = tcim::Tensor::CreateHostTensor(resized_yuv_info);
    } else {
      resized_yuv_tensor = tcim::Tensor::CreateDeviceTensor(resized_yuv_info);
    }
    tcim::ImageOps::RectRoi roi(0, 0, width, height);
    tcim::ImageOps::Resizer::RunOption run_opt(roi);
    run_opt.interp_mode = tcim::ImageOps::Resizer::EnInterpMode::Nearest;
    img_resizer.resizer->Run(yuv_tensor, resized_yuv_tensor, run_opt, true);

#ifdef SAVE_IMGS
    auto decoded_tensor = yuv_tensor.ToHost(true);
    auto resized_tensor = resized_yuv_tensor.ToHost(true);

    std::string decoded_file_name = "decoded_" + std::to_string(fid);
    SaveImgs(height, width, decoded_tensor.Data(), decoded_file_name);
    LOG_INFO << "save the decoded image, file_path:" << decoded_file_name;

    std::string resized_file_name = "resized_" + std::to_string(fid);
    SaveImgs(md_heigth, md_width, resized_tensor.Data(), resized_file_name);
    LOG_INFO << "save the resized image, file_path:" << resized_file_name;
#endif
#endif

    count++;
    decoder.ReleaseBuf();

    // push input data to det queue
    TaskInfo task_info;
    task_info.req_id = fid++;
#ifdef RESIZER
    task_info.image = resized_yuv_tensor.Buffer();
#else
    task_info.image = yuv_tensor.Buffer();
#endif

    bool print_flag = true;
    while (1) {
      std::unique_lock<std::mutex> lock(qout.mutex);
      // check if det queue is too full
      int size = qout.queue.size();
      if (size <= 10) {
        qout.queue.push(task_info);
        LOG_DEBUG << "decoder pull data req_id " << task_info.req_id << ", queue size " << size;
        qout.cond.notify_all();
        lock.unlock();
        print_flag = true;
        break;
      }
      if (print_flag) {
        LOG_WARN << "det queue size " << size << " exceed 10, get frame suspended.";
        print_flag = false;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  decoder.Close();
  LOG_INFO << "<=== GetFrame thread exit. " << count << " frames received.";
}

// define detect thread
void detect(InferInfo& infer_info, TaskQueue& qin, TaskQueue& qout, Barrier& barrier) {
  YoloV5 yolov5;
  int32_t dyn_info[10] = {0, 0, 1080, 1920, 360, 640, 140, 0, 140, 0};
  // wait until all threads ready
  barrier.barrier();
  int count = 0;
  auto& module = infer_info.module;
  std::string image_input_name = "images";
  std::string dyn_info_name = "dyn_info";
  auto& input_infos = module.GetInputInfoMap();
  auto& output_infos = module.GetOutputInfoMap();
  LOG_INFO << "detect thread " << infer_info.id << " infer start...";

  // detect loop
  while (true) {
    // get data from the task queue
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task_info = qin.queue.front();
    if (task_info.is_end) {
      LOG_INFO << "===> detect thread EOS received.";
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();
      lock_in.unlock();
      break;
    }
    qin.queue.pop();
    lock_in.unlock();

    std::map<std::string, tcim::Tensor> input_map;
    std::map<std::string, tcim::Tensor> output_map;

    // prepare input
    auto input_info = input_infos[image_input_name];
    if (task_info.image.Device() == tcim::CPU) {
      input_info = input_info.AsContiguous();
    }
    input_map[image_input_name] = tcim::Tensor(input_info, task_info.image);
    auto it = input_infos.find(dyn_info_name);
    if (it != input_infos.end()) {
      input_map[dyn_info_name] = tcim::Tensor::CreateHostTensor(it->second.AsContiguous());
      memcpy(input_map[dyn_info_name].Data(), dyn_info, 10 * sizeof(int32_t));
    }

    // prepare output
    for (auto& output_info : output_infos) {
      auto info = output_info.second.AsContiguous().AsType(tcim::FLOAT32);
      output_map[output_info.first] = tcim::Tensor::CreateHostTensor(info);
    }

    auto start = GET_TIME();

    // set input to the module
    for (auto& input : input_map) {
      module.SetInput(input.first, input.second);
    }

    // run and sync
    module.Run();
    module.Sync();

    // get output and push to the output queue
    for (auto& output : output_map) {
      auto output_tensor = module.GetOutput(output.first);
      output_tensor.CastTo(output.second);
    }

    auto end = GET_TIME();
    auto cost = GET_COST(start, end) / 1000.0;
    LOG_DEBUG << "detect thread " << infer_info.id << " run sample " << task_info.req_id
              << " end. cost " << cost << " ms.";

    count++;

    // postprocess
    tcim::Tensor tensor;
    if (task_info.image.Device() == tcim::HDPL) {
      auto info = input_map[image_input_name].Info().AsContiguous();
      tensor = tcim::Tensor::CreateHostTensor(info);
      input_map[image_input_name].CopyTo(tensor);
    } else {
      tensor = input_map[image_input_name];
    }
    cv::Mat nv12(1080 * 3 / 2, 1920, CV_8UC1, tensor.Data());
    cv::Mat rgb;
    cv::Mat bgr;
    cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    cv::cvtColor(rgb, bgr, cv::COLOR_BGR2RGB);

    std::vector<DetectOutput> outputs;
    for (auto& output : output_map) {
      DetectOutput out;
      out.data = (float*)output.second.Data();
      auto& shape = output.second.Info().Shape();
      out.num_anchors = shape[1] * shape[2] * shape[3];
      out.stride = 640 / shape[2];
      outputs.emplace_back(out);
    }

    auto detections = yolov5.postprocess(bgr, outputs);

    // print and draw
    printf("detect num: %d\n", (int)detections.size());
    for (const auto& detection : detections) {
      // printf("box[%d, %d, %d, %d], conf:%f, cls:%d\n", detection.box.x1, detection.box.y1,
      //        detection.box.x2, detection.box.y2, detection.conf, detection.cls);
      cv::rectangle(bgr, cv::Point(detection.box.x1, detection.box.y1),
                    cv::Point(detection.box.x2, detection.box.y2), cv::Scalar(0, 0, 255), 2);
    }
    fs::path file_path(std::to_string(task_info.req_id) + ".jpg");
    fs::path result_path("demo_results");
    if (!fs::exists(result_path)) {
      fs::create_directory("demo_results");
    }
    fs::path result_file = result_path / file_path.filename();
    cv::imwrite(result_file.string().c_str(), bgr);
    LOG_DEBUG << "demo results saved to " << result_file.string();

    // send to classify threads
    for (const auto& detection : detections) {
      ObjInfo obj;
      obj.det = detection;
      task_info.objs.push_back(obj);
    }
    bool print_flag = true;
    while (1) {
      std::unique_lock<std::mutex> lock(qout.mutex);
      // check if cls queue is too full
      int size = qout.queue.size();
      if (size <= 10) {
        qout.queue.push(task_info);
        LOG_DEBUG << "detect push task req_id " << task_info.req_id
                  << ", queue size " << size;
        qout.cond.notify_all();
        lock.unlock();
        print_flag = true;
        break;
      }
      if (print_flag) {
        LOG_WARN << "classify queue size " << size << " exceed 10, detect suspended.";
        print_flag = false;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  LOG_INFO << "<=== detect thread " << infer_info.id << " completed. "
           << count << " sampels tested.";
}


// define classify thread
void classify(InferInfo& infer_info, TaskQueue& qin, TaskQueue& qout, Barrier& barrier) {
  Resnet50 resnet50;
  int count = 0;
  auto& module = infer_info.module;
  std::string image_input_name = "input.1";
  std::string dyn_info_name = "dyn_info";
  auto& input_infos = module.GetInputInfoMap();
  auto& output_infos = module.GetOutputInfoMap();
  LOG_INFO << "classify thread " << infer_info.id << " infer start...";

  // classify loop
  while (true) {
    // get data from the task queue
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task_info = qin.queue.front();
    if (task_info.is_end) {
      LOG_INFO << "===> classify thread EOS received.";
      lock_in.unlock();
      break;
    }
    qin.queue.pop();
    lock_in.unlock();

    std::map<std::string, tcim::Tensor> input_map;
    std::map<std::string, tcim::Tensor> output_map;
    int det_cnt = 0;

    for (auto& obj : task_info.objs) {
      // prepare input
      auto input_info = input_infos[image_input_name];
      if (task_info.image.Device() == tcim::CPU) {
        input_info = input_info.AsContiguous();
      }
      input_map[image_input_name] = tcim::Tensor(input_info, task_info.image);
      auto it = input_infos.find(dyn_info_name);
      if (it != input_infos.end()) {
        input_map[dyn_info_name] = tcim::Tensor::CreateHostTensor(it->second.AsContiguous());
        auto dyn_info = static_cast<int32_t*>(input_map[dyn_info_name].Data());
        // roi crop [y1, x1, h, w]
        dyn_info[0] = TO_EVEN(obj.det.box.y1);
        dyn_info[1] = TO_EVEN(obj.det.box.x1);
        dyn_info[2] = TO_EVEN(obj.det.box.h());
        dyn_info[3] = TO_EVEN(obj.det.box.w());
        // resize [H, W]
        dyn_info[4] = resnet50.input_sizes_[1];
        dyn_info[5] = resnet50.input_sizes_[0];
        // pad [top, left, bottom, right]
        dyn_info[6] = 0;
        dyn_info[7] = 0;
        dyn_info[8] = 0;
        dyn_info[9] = 0;
      }

      // prepare output
      for (auto& output_info : output_infos) {
        auto info = output_info.second.AsContiguous().AsType(tcim::FLOAT32);
        output_map[output_info.first] = tcim::Tensor::CreateHostTensor(info);
      }

      auto start = GET_TIME();

      // set input to the module
      for (auto& input : input_map) {
        module.SetInput(input.first, input.second);
      }

      // run and sync
      module.Run();
      module.Sync();

      // get output and push to the output queue
      for (auto& output : output_map) {
        auto output_tensor = module.GetOutput(output.first);
        output_tensor.CastTo(output.second);
      }

      auto end = GET_TIME();
      auto cost = GET_COST(start, end) / 1000.0;
      LOG_DEBUG << "classify thread " << infer_info.id << " run sample " << task_info.req_id << " obj "
                <<  det_cnt << " end. cost " << cost << " ms.";
      det_cnt++;

      auto cls = resnet50.postprocess(static_cast<float*>(output_map.begin()->second.Data()), 1000);

      // print
      printf("sample %lld box[%d, %d, %d, %d], det[conf:%f, cls:%d], cls[id:%d, conf:%f, lable:[%s]]\n",
             task_info.req_id, obj.det.box.x1, obj.det.box.y1, obj.det.box.x2, obj.det.box.y2,
             obj.det.conf, obj.det.cls, cls[0].index, cls[0].conf, Imagenet::GetLabel(cls[0].index).c_str());
    }
    count++;
  }
  LOG_INFO << "<=== classify thread " << infer_info.id << " completed. " << count << " sampels tested.";
}


int main(int argc, char **argv)
{
  std::string det_model_path = "yolov5s.hmm";
  std::string cls_model_path = "resnet50.hmm";
  std::string stream_path = "../../data/1080P_traffic_4s.h264";
  int det_thread_num = 1;
  int cls_thread_num = 2;
  size_t frame_limit = 0;

  std::vector<std::thread> threads;
  TaskQueue q_det;
  TaskQueue q_cls;
  TaskQueue q_out;

  if (auto platform = std::getenv("HDPL_PLATFORM")) {
    if (!strcmp(platform, "ISIM")) {
      det_thread_num = 1;
      cls_thread_num = 1;
      frame_limit = 2;
      LOG_WARN << "det_thread_num set to " << det_thread_num << " while HDPL_PLATFORM=ISIM";
      LOG_WARN << "cls_thread_num set to " << cls_thread_num << " while HDPL_PLATFORM=ISIM";
      LOG_WARN << "frame_limit set to " << frame_limit << " while HDPL_PLATFORM=ISIM";
    }
  }

  auto postfix = stream_path.substr(stream_path.size() - 5);
  std::string format;
  if (postfix == ".h264") {
    format = "H264";
  } else if (postfix == ".h265") {
    format = "H265";
  } else {
    LOG_ERROR << "file format not supported: " << stream_path;
    exit(-1);
  }

  // create infer threads
  Barrier barrier(det_thread_num);
  std::vector<InferInfo> det_infer_infos(det_thread_num);
  std::vector<InferInfo> cls_infer_infos(cls_thread_num);

  auto wm = tcim::Module::WeightManager::CreateWeightManager(0);
  for (int i = 0; i < det_thread_num; i++) {
    InferInfo infer_info;
    if (infer_info.module.Load(det_model_path, wm)) {
      LOG_ERROR << "load model fail: " << det_model_path;
      exit(-1);
    }
    LOG_INFO << "thread " << i << " model loaded: " << det_model_path;
    infer_info.id = i;
    det_infer_infos[i] = infer_info;
    threads.emplace_back(std::thread(&detect, std::ref(det_infer_infos[i]), std::ref(q_det),
                         std::ref(q_cls), std::ref(barrier)));
  }
  for (int i = 0; i < cls_thread_num; i++) {
    InferInfo infer_info;
    if (infer_info.module.Load(cls_model_path, wm)) {
      LOG_ERROR << "load model fail: " << cls_model_path;
      exit(-1);
    }
    LOG_INFO << "thread " << i << " model loaded: " << cls_model_path;
    infer_info.id = i;
    cls_infer_infos[i] = infer_info;
    threads.emplace_back(std::thread(&classify, std::ref(cls_infer_infos[i]), std::ref(q_cls),
                         std::ref(q_out), std::ref(barrier)));
  }
  wm.~WeightManager();
  barrier.wait();

  VideoDecoder decoder(format, "NV12M");
  Barrier barrier2(2);
  PushStreamInfo stream_info = {stream_path, frame_limit};
#ifdef RESIZER
  decoder.SetModelInfo(1920, 1080);  // yolov5s input shape (width, height)
#endif
  // create push stream thread
  threads.emplace_back(std::thread(&PushStream, std::ref(decoder), std::ref(stream_info), std::ref(barrier2)));
  // create rcv frame thread
  threads.emplace_back(std::thread(&GetFrame, std::ref(decoder), std::ref(q_det), std::ref(barrier2)));
  barrier2.wait();

  for (auto & t: threads) {
    t.join();
  }

  return 0;
}

