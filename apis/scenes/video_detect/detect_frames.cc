#include "detect_frames.h"

int GetInputInfoMap(PooledModule *module,
                    std::map<std::string, tcim::TensorInfo> &input_info_map) {
  int input_num = module->GetInputNum();
  LOG_INFO("Count of Input: {}", input_num);
  for (int idx = 0; idx < input_num; idx++) {
    auto input_name = module->GetInputName(idx);
    auto input_info = module->GetInputInfo(input_name, false);
    LOG_INFO("Input[{}] info: {}", input_name, TensorInfo2Str(input_info));
    input_info_map[input_name] = input_info;
  }
  return 0;
}

int GetOutputInfoMap(PooledModule *module,
                     std::map<std::string, tcim::TensorInfo> &output_info_map) {
  int output_num = module->GetOutputNum();
  LOG_INFO("Count of Output: {}", output_num);
  for (int idx = 0; idx < output_num; idx++) {
    auto output_name = module->GetOutputName(idx);
    auto output_info = module->GetOutputInfo(output_name, false);
    LOG_INFO("Output[{}] info: {}", output_name, TensorInfo2Str(output_info));
    output_info_map[output_name] = output_info;
  }
  return 0;
}

void detect(InferInfo &infer_info, TaskQueue &qin, TaskQueue &qout,
            TaskQueue &qout_enc, bool classify_task, Barrier &barrier) {
  YoloV5 yolov5;
  int ret = 0;
  int count = 0;

  // {cropX, cropY, crop height, crop width, resize heigth, resize width, pad
  // top, pad left, pad bottom, pad right}
  int32_t dyn_info[10] = {0, 0, 1080, 1920, 360, 640, 140, 0, 140, 0};
  std::string image_input_name = "images";
  std::string dyn_info_name = "dyn_info";

  PooledModule *module = infer_info.module;
  std::map<std::string, tcim::TensorInfo> input_infos;
  std::map<std::string, tcim::TensorInfo> output_infos;
  GetInputInfoMap(module, input_infos);
  GetOutputInfoMap(module, output_infos);

  // prepare dyn_info input
  // because the size of the decoded output image is fixed, the dyn_info is
  // determined and can be created in advance.
  std::map<std::string, tcim::Tensor> input_map;
  auto it = input_infos.find(dyn_info_name);
  if (it != input_infos.end()) {
    auto dyn_host_tensor =
      tcim::Tensor::CreateHostTensor(it->second.AsContiguous());
    memcpy(dyn_host_tensor.Data(), dyn_info, 10 * sizeof(int32_t));
    auto host_mem_size = dyn_host_tensor.MemSize();
    auto dyn_dev_buf = tcim::Buffer::CreateDeviceBuffer(
      dyn_host_tensor.MemSize(), DEVICE_ID, "", "reserved");
    auto dyn_dev_tensor = tcim::Tensor(it->second, dyn_dev_buf);
    dyn_host_tensor.CopyTo(dyn_dev_tensor);
    input_map[dyn_info_name] = dyn_dev_tensor;
  }

  // prepare outputs
  std::map<std::string, tcim::Tensor> output_map_i8;
  std::map<std::string, tcim::Tensor> output_map_f32;
  for (auto &output_info : output_infos) {
    auto info = output_info.second.AsContiguous();
    auto info_f32 = info.AsType(tcim::FLOAT32);
    output_map_i8[output_info.first] = tcim::Tensor::CreateHostTensor(info);
    output_map_f32[output_info.first] =
      tcim::Tensor::CreateHostTensor(info_f32);
  }

  // wait until all threads ready
  barrier.barrier();
  LOG_INFO("detect thread, moudle {} infer start, current mem {} MB...",
           infer_info.id, getCurrentMemoryUsage());

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
#ifdef ENC_TASK
      std::unique_lock<std::mutex> lock_enc_out(qout_enc.mutex);
      qout_enc.queue.push(task_info);
      qout_enc.cond.notify_all();
      lock_enc_out.unlock();
#endif
      lock_in.unlock();
      break;
    }
    // qin.queue.pop();
    lock_in.unlock();

#ifdef RK_DECODER
    // copy rk decoded yuv data to device
    auto input_info = input_infos[image_input_name];
    tcim::Tensor decoded_dev_tensor =
      tcim::Tensor::CreateDeviceTensor(input_info);
    tcim::Buffer y_buf = tcim::Buffer::CreateHostBuffer(
      task_info.y_buf_size, (void *)task_info.y_buf);
    tcim::Buffer uv_buf = tcim::Buffer::CreateHostBuffer(
      task_info.uv_buf_size, (void *)task_info.uv_buf);
    y_buf.CopyTo(decoded_dev_tensor.Buffer(), task_info.y_buf_size);
    uv_buf.CopyTo(decoded_dev_tensor.Buffer(), task_info.uv_buf_size, 0,
                  task_info.y_buf_size);
    // release rk decoder buffer
    mpp_frame_deinit(task_info.frame);
    delete task_info.frame;
    task_info.frame = nullptr;

    task_info.image = decoded_dev_tensor.Buffer();
    // prepare image input
    input_map[image_input_name] = decoded_dev_tensor;

#ifdef DUMP_RK_DECODED_DATA
    std::string decoded_file_name = "rk_" + std::to_string(task_info.req_id);
    auto decoded_host_tensor = decoded_dev_tensor.ToHost(true);
    SaveImgs(1080, 1920, decoded_host_tensor.Data(), "decoder_results",
             decoded_file_name);
    LOG_INFO("save the decoded image, file_path: {}", decoded_file_name);
#endif  // DUMP_RK_DECODED_DATA
#else   // !RK_DECODER

    auto input_device = task_info.image.Device();
    // prepare image input
    auto input_info = input_infos[image_input_name];
    if (input_device == tcim::Device::CPU) {
      input_info = input_info.AsContiguous();
    }
    input_map[image_input_name] = tcim::Tensor(input_info, task_info.image);
#endif  // RK_DECODER

    auto start = GET_TIME();
    ret = module->Infer(input_map, output_map_i8);
    if (ret != 0) {
      LOG_ERROR("detect thread {} infer sample {} error, skip.", infer_info.id,
                task_info.req_id);
      continue;
    }
    // convert output datatype from int8 to float32
    for (auto &output : output_map_i8) {
      output.second.CastTo(output_map_f32[output.first]);
    }
    auto end = GET_TIME();
    auto cost = GET_COST(start, end) / 1000.0;
    LOG_INFO("detect thread {} run sample {} end. cost {} ms.", infer_info.id,
             task_info.req_id, cost);
    count++;

    // yolov5s postprocess
    std::vector<DetectOutput> outputs;
    for (auto &output : output_map_f32) {
      DetectOutput out;
      out.data = (float *)output.second.Data();
      auto &shape = output.second.Info().Shape();
      out.num_anchors = shape[1] * shape[2] * shape[3];
      out.stride = 640 / shape[2];
      outputs.emplace_back(out);
    }

    auto detections = yolov5.postprocess(task_info.frame_height,
                                         task_info.frame_width, outputs);

    {
      // delayed pop reduces the image data in the device cache.
      std::unique_lock<std::mutex> lock_in(qin.mutex);
      qin.queue.pop();
      lock_in.unlock();
    }

#ifdef GEN_IMGS
#ifdef RK_DECODER
    // copy input image from device to host
    tcim::Tensor tensor = input_map[image_input_name].ToHost(true);
#else   // !RK_DECODER
    tcim::Tensor tensor;
    if (input_device == tcim::Device::CPU) {
      tensor = input_map[image_input_name];
    } else {
      tensor = input_map[image_input_name].ToHost(true);
    }
#endif  // RK_DECODER

    // generate detection results
    cv::Mat nv12(1080 * 3 / 2, 1920, CV_8UC1, tensor.Data());
    cv::Mat rgb;
    cv::Mat bgr;
    cv::cvtColor(nv12, rgb, cv::COLOR_YUV2RGB_NV12);
    cv::cvtColor(rgb, bgr, cv::COLOR_BGR2RGB);
    // print and draw
    LOG_INFO("detect thread {} sample {} detect {} targets.", infer_info.id,
             task_info.req_id, detections.size());
    for (const auto &detection : detections) {
      cv::rectangle(bgr, cv::Point(detection.box.x1, detection.box.y1),
                    cv::Point(detection.box.x2, detection.box.y2),
                    cv::Scalar(0, 0, 255), 2);
    }
    // save as image
    fs::path file_path(std::to_string(infer_info.id) + "_" +
                       std::to_string(task_info.req_id) + ".jpg");
    fs::path result_path("demo_results");
    if (!fs::exists(result_path)) {
      fs::create_directory("demo_results");
    }
    fs::path result_file = result_path / file_path.filename();
    std::string result_path_str = result_file.string();
    cv::imwrite(result_path_str.c_str(), bgr);

#ifdef ENC_TASK
    // convert rectangle buffer to yuv format
    int size = 1080 * 1920 * 3;
    size_t yuv_total_size = size / 2;
    tcim::Buffer rect_buf = tcim::Buffer::CreateHostBuffer(yuv_total_size);
    ImageProc::BgrToRgb((int8_t *)(bgr.data), bgr.rows, bgr.cols);
    cv::Mat img_yuv;
    cv::cvtColor(bgr, img_yuv, cv::COLOR_RGB2YUV_I420);
    ImageProc::I420To420sp((uint8_t *)rect_buf.Data(), (uint8_t *)img_yuv.data,
                           size);

    // construct encoder task and push into task queue
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
#endif  // ENC_TASK
#endif  // GEN_IMGS

    if (!classify_task) {
      continue;
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

void classify(InferInfo &infer_info, TaskQueue &qin, Barrier &barrier) {
  Resnet50 resnet50;
  int ret = 0;
  int count = 0;

  std::string image_input_name = "input.1";
  std::string dyn_info_name = "dyn_info";

  PooledModule *module = infer_info.module;
  std::map<std::string, tcim::TensorInfo> input_infos;
  std::map<std::string, tcim::TensorInfo> output_infos;
  GetInputInfoMap(module, input_infos);
  GetOutputInfoMap(module, output_infos);

  // prepare dyn_info input, pre-allocate host buffer
  std::map<std::string, tcim::Tensor> input_map;
  int32_t *dyn_info = nullptr;
  auto it = input_infos.find(dyn_info_name);
  if (it != input_infos.end()) {
    input_map[dyn_info_name] =
      tcim::Tensor::CreateHostTensor(it->second.AsContiguous());
    dyn_info = static_cast<int32_t *>(input_map[dyn_info_name].Data());
  }

  // prepare outputs
  std::map<std::string, tcim::Tensor> output_map_i8;
  std::map<std::string, tcim::Tensor> output_map_f32;
  for (auto &output_info : output_infos) {
    auto info = output_info.second.AsContiguous();
    auto info_f32 = info.AsType(tcim::FLOAT32);
    output_map_i8[output_info.first] = tcim::Tensor::CreateHostTensor(info);
    output_map_f32[output_info.first] =
      tcim::Tensor::CreateHostTensor(info_f32);
  }

  barrier.barrier();
  LOG_INFO("classify thread, module {} infer start, current mem {} MB ...",
           infer_info.id, getCurrentMemoryUsage());

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

    int det_cnt = 0;
    for (auto &obj : task_info.objs) {
      // prepare input
      auto input_info = input_infos[image_input_name];
      if (task_info.image.Device() == tcim::Device::CPU) {
        input_info = input_info.AsContiguous();
      }
      input_map[image_input_name] = tcim::Tensor(input_info, task_info.image);

      if (dyn_info != nullptr) {
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

      auto start = GET_TIME();
      ret = module->Infer(input_map, output_map_i8);
      if (ret != 0) {
        LOG_ERROR("detect thread {} infer sample {} error, skip.",
                  infer_info.id, task_info.req_id);
        continue;
      }
      // convert output datatype from int8 to float32
      for (auto &output : output_map_i8) {
        output.second.CastTo(output_map_f32[output.first]);
      }
      auto end = GET_TIME();
      auto cost = GET_COST(start, end) / 1000.0;
      LOG_INFO("classify thread {} run sample {} obj {} end. cost {} ms.",
               infer_info.id, task_info.req_id, det_cnt, cost);
      det_cnt++;

      auto cls = resnet50.postprocess(
        static_cast<float *>(output_map_f32.begin()->second.Data()), 1000);

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
