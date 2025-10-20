#include "detect_frames.h"

// define detect thread
void detect(InferInfo &infer_info, TaskQueue &qin, TaskQueue &qout,
            TaskQueue &qout_enc, Barrier &barrier) {
  YoloV5 yolov5;
  // {cropX, cropY, crop height, crop width, resize heigth, resize width, pad
  // top, pad left, pad bottom, pad right}
  int32_t dyn_info[10] = {0, 0, 1080, 1920, 360, 640, 140, 0, 140, 0};

  int count = 0;
  auto &module = infer_info.module;
  std::string image_input_name = "images";
  std::string dyn_info_name = "dyn_info";
  auto &input_infos = module.GetInputInfoMap();
  auto &output_infos = module.GetOutputInfoMap();

  // wait until all threads ready
  barrier.barrier();
  LOG_INFO("detect thread, moudle {} infer start...", infer_info.id);

  // detect loop
  while (true) {
    // get data from the task queue
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task_info = qin.queue.front();
    if (task_info.is_end) {
      LOG_INFO("===> detect thread EOS received.");
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();
      std::unique_lock<std::mutex> lock_enc_out(qout_enc.mutex);
      qout_enc.queue.push(task_info);
      qout_enc.cond.notify_all();
      lock_enc_out.unlock();

      lock_in.unlock();
      break;
    }
    qin.queue.pop();
    lock_in.unlock();

    std::map<std::string, tcim::Tensor> input_map;
    std::map<std::string, tcim::Tensor> output_map;

#ifdef RK_DECODER
#ifdef DUMP_RK_DECODED_DATA
    std::string decoded_file_name = "rk_" + std::to_string(task_info.req_id);
    SaveImgs(1080, 1920, task_info.buffer.get(), "decoder_results",
             decoded_file_name);
    LOG_INFO("save the decoded image, file_path: {}", decoded_file_name);
#endif  // DUMP_RK_DECODED_DATA

    // create a tensor for rk decoded data
    auto input_info = input_infos[image_input_name];
    tcim::Tensor decoded_host_tensor;
    if (input_info.MemSize() > task_info.buffer_length) {
      decoded_host_tensor = tcim::Tensor::CreateHostTensor(input_info);
      memcpy(decoded_host_tensor.Data(), task_info.buffer.get(),
             task_info.buffer_length);
    } else {
      decoded_host_tensor = tcim::Tensor::CreateHostTensor(
          input_info, input_info.MemSize(), task_info.buffer.get());
    }
    task_info.image = decoded_host_tensor.Buffer();

    // prepare image input
    input_map[image_input_name] = decoded_host_tensor;
#else   // !RK_DECODER

    auto input_device = task_info.image.Device();
    // prepare image input
    auto input_info = input_infos[image_input_name];
    if (input_device == tcim::Device::CPU) {
      input_info = input_info.AsContiguous();
    }
    input_map[image_input_name] = tcim::Tensor(input_info, task_info.image);
#endif  // RK_DECODER

    // prepare dyn_info input
    auto it = input_infos.find(dyn_info_name);
    if (it != input_infos.end()) {
      input_map[dyn_info_name] =
          tcim::Tensor::CreateHostTensor(it->second.AsContiguous());
      memcpy(input_map[dyn_info_name].Data(), dyn_info, 10 * sizeof(int32_t));
    }

    // prepare output
    for (auto &output_info : output_infos) {
      auto info = output_info.second.AsContiguous().AsType(tcim::FLOAT32);
      output_map[output_info.first] = tcim::Tensor::CreateHostTensor(info);
    }

    auto start = GET_TIME();

    // set input to the module
    for (auto &input : input_map) {
      module.SetInput(input.first, input.second);
    }

    // run and sync
    module.Run();
    module.Sync();

    // get output and push to the output queue
    for (auto &output : output_map) {
      auto output_tensor = module.GetOutput(output.first);
      output_tensor.CastTo(output.second);
    }

    auto end = GET_TIME();
    auto cost = GET_COST(start, end) / 1000.0;
    LOG_INFO("detect thread {} run sample {} end. cost {} ms.", infer_info.id,
             task_info.req_id, cost);

    count++;

    // postprocess
#ifdef RK_DECODER
    tcim::Tensor tensor = input_map[image_input_name];
#else   // !RK_DECODER
    tcim::Tensor tensor;
    if (input_device == tcim::Device::CPU) {
      tensor = input_map[image_input_name];
    } else {
      auto info = input_map[image_input_name].Info().AsContiguous();
      tensor = tcim::Tensor::CreateHostTensor(info);
      input_map[image_input_name].CopyTo(tensor);
    }
#endif  // RK_DECODER

    cv::Mat nv12(1080 * 3 / 2, 1920, CV_8UC1, tensor.Data());
    cv::Mat rgb;
    cv::Mat bgr;
    cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    cv::cvtColor(rgb, bgr, cv::COLOR_BGR2RGB);

    std::vector<DetectOutput> outputs;
    for (auto &output : output_map) {
      DetectOutput out;
      out.data = (float *)output.second.Data();
      auto &shape = output.second.Info().Shape();
      out.num_anchors = shape[1] * shape[2] * shape[3];
      out.stride = 640 / shape[2];
      outputs.emplace_back(out);
    }

    auto detections = yolov5.postprocess(bgr, outputs);

    // print and draw
    LOG_INFO("detect num: {}", detections.size());
    for (const auto &detection : detections) {
      // LOG_INFO("box[{}, {}, {}, {}], conf:{}, cls:{}", detection.box.x1,
      // detection.box.y1,
      //        detection.box.x2, detection.box.y2, detection.conf,
      //        detection.cls);
      cv::rectangle(bgr, cv::Point(detection.box.x1, detection.box.y1),
                    cv::Point(detection.box.x2, detection.box.y2),
                    cv::Scalar(0, 0, 255), 2);
    }
    fs::path file_path(std::to_string(infer_info.id) + "_" +
                       std::to_string(task_info.req_id) + ".jpg");
    fs::path result_path("demo_results");
    if (!fs::exists(result_path)) {
      fs::create_directory("demo_results");
    }
    fs::path result_file = result_path / file_path.filename();
    cv::imwrite(result_file.string().c_str(), bgr);
    LOG_DEBUG("demo results saved to {}", result_file.string());

    int size = 1080 * 1920 * 3;
    size_t yuv_total_size = size / 2;
    tcim::Buffer rect_buf = tcim::Buffer::CreateHostBuffer(yuv_total_size);
    ImageProc::BgrToRgb((int8_t *)(bgr.data), bgr.rows, bgr.cols);
    cv::Mat img_yuv;
    cv::cvtColor(bgr, img_yuv, cv::COLOR_RGB2YUV_I420);
    ImageProc::I420To420sp((uint8_t *)rect_buf.Data(), (uint8_t *)img_yuv.data,
                           size);

    TaskInfo enc_task;
    enc_task.req_id = task_info.req_id;
    enc_task.image = rect_buf;
    {
      std::unique_lock<std::mutex> lock_enc(qout_enc.mutex);
      int size = qout_enc.queue.size();
      qout_enc.queue.push(enc_task);
      LOG_INFO("detect push enc task, req_id {}, queue size {}.",
               task_info.req_id, size);
      qout_enc.cond.notify_all();
      lock_enc.unlock();
    }

    // send to classify threads
    for (const auto &detection : detections) {
      ObjInfo obj;
      obj.det = detection;
      task_info.objs.push_back(obj);
    }
    bool print_flag = true;
    while (1) {
      std::unique_lock<std::mutex> lock(qout.mutex);
      // check if cls queue is too full
      int size = qout.queue.size();
      if (size <= INFERENCE_QUEUE_SIZE) {
        qout.queue.push(task_info);
        LOG_INFO("detect push task req_id {} obj num {}, queue size {}.",
                 task_info.req_id, task_info.objs.size(), size);
        qout.cond.notify_all();
        lock.unlock();
        print_flag = true;
        break;
      }
      if (print_flag) {
        LOG_WARNING("classify queue size {} exceed 10, detect suspended.",
                    size);
        print_flag = false;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }
  LOG_INFO("<=== detect thread {} completed. {} sampels tested.", infer_info.id,
           count);
}

// define classify thread
void classify(InferInfo &infer_info, TaskQueue &qin, Barrier &barrier) {
  Resnet50 resnet50;
  int count = 0;
  auto &module = infer_info.module;
  std::string image_input_name = "input.1";
  std::string dyn_info_name = "dyn_info";
  auto &input_infos = module.GetInputInfoMap();
  auto &output_infos = module.GetOutputInfoMap();

  barrier.barrier();
  LOG_INFO("classify thread, module {} infer start...", infer_info.id);

  // classify loop
  while (true) {
    // get data from the task queue
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task_info = qin.queue.front();
    if (task_info.is_end) {
      LOG_INFO("===> classify thread EOS received.");
      lock_in.unlock();
      break;
    }
    qin.queue.pop();
    lock_in.unlock();

    LOG_INFO("classify thread receives task {}, obj num {}.", task_info.req_id,
             task_info.objs.size());

    std::map<std::string, tcim::Tensor> input_map;
    std::map<std::string, tcim::Tensor> output_map;
    int det_cnt = 0;

    for (auto &obj : task_info.objs) {
      // prepare input
      auto input_info = input_infos[image_input_name];
      if (task_info.image.Device() == tcim::Device::CPU) {
        input_info = input_info.AsContiguous();
      }
      input_map[image_input_name] = tcim::Tensor(input_info, task_info.image);

      auto it = input_infos.find(dyn_info_name);
      if (it != input_infos.end()) {
        input_map[dyn_info_name] =
            tcim::Tensor::CreateHostTensor(it->second.AsContiguous());
        auto dyn_info = static_cast<int32_t *>(input_map[dyn_info_name].Data());
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
      for (auto &output_info : output_infos) {
        auto info = output_info.second.AsContiguous().AsType(tcim::FLOAT32);
        output_map[output_info.first] = tcim::Tensor::CreateHostTensor(info);
      }

      auto start = GET_TIME();

      // set input to the module
      for (auto &input : input_map) {
        module.SetInput(input.first, input.second);
      }

      // run and sync
      module.Run();
      module.Sync();

      // get output and push to the output queue
      for (auto &output : output_map) {
        auto output_tensor = module.GetOutput(output.first);
        output_tensor.CastTo(output.second);
      }

      auto end = GET_TIME();
      auto cost = GET_COST(start, end) / 1000.0;
      LOG_INFO("classify thread {} run sample {} obj {} end. cost {} ms.",
               infer_info.id, task_info.req_id, det_cnt, cost);
      det_cnt++;

      auto cls = resnet50.postprocess(
          static_cast<float *>(output_map.begin()->second.Data()), 1000);

      // print
      LOG_INFO(
          "sample {} box[{}, {}, {}, {}], det[conf:{}, cls:{}], cls[id:{}, "
          "conf:{}, lable:[{}]]",
          task_info.req_id, obj.det.box.x1, obj.det.box.y1, obj.det.box.x2,
          obj.det.box.y2, obj.det.conf, obj.det.cls, cls[0].index, cls[0].conf,
          Imagenet::GetLabel(cls[0].index));
    }
    count++;
  }
  LOG_INFO("<=== classify thread {} completed. {} sampels tested.",
           infer_info.id, count);
}
