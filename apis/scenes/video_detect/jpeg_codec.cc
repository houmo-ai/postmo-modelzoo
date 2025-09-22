#include "jpeg_codec.h"

void JpegCodec::EncodeImage(JpegEncoder &encoder, char *yuv_buffer,
                            int iteration, TaskQueue &qout, Barrier &barrier) {
  LOG_INFO("===> EncodeImage thread start, ready to push {} images.",
           iteration);

  int ret = 0;

  auto width = encoder.GetWidth();
  auto height = encoder.GetHeight();
  const int y_size = width * height;

  char *y_buf = yuv_buffer;
  char *uv_buf = (yuv_buffer + y_size);

  FrameDevice device = HDPL;
  EncodedData jpeg_data = {0};

  barrier.barrier();

  for (int i = 0; i <= iteration; i++) {
    if (i == iteration) {
      TaskInfo task_info;
      task_info.req_id = i;
      task_info.is_end = true;
      std::unique_lock<std::mutex> lock(qout.mutex);
      qout.queue.push(task_info);
      LOG_INFO("Jpeg encoder push eos data {}.", task_info.req_id);
      qout.cond.notify_all();
      lock.unlock();
      break;
    }

    ret = encoder.PushData(y_buf, uv_buf, 1);
    if (ret != 0) {
      LOG_ERROR("Push data into JPEG encoder failed: {}.", ret);
      ret = -1;
      break;
    }

    ret = encoder.PullData(jpeg_data, device);
    auto device_buf = tcim::Buffer::CreateDeviceBuffer(
        jpeg_data.data, jpeg_data.len, DEVICE_ID, "");
    auto host_buf = tcim::Buffer::CreateHostBuffer(jpeg_data.len);
    device_buf.CopyTo(host_buf, jpeg_data.len);

    encoder.ReleaseBuf();

    // push encoded data to decoder queue
    TaskInfo task_info;
    task_info.req_id = i;
    task_info.image = host_buf;

    bool print_flag = true;
    while (true) {
      std::unique_lock<std::mutex> lock(qout.mutex);
      // check if det queue is too full
      int size = qout.queue.size();
      if (size <= DECODER_QUEUE_SIZE) {
        qout.queue.push(task_info);
        LOG_INFO("Jpeg encoder push encoded data {}, queue size {}.",
                 task_info.req_id, size);
        qout.cond.notify_all();
        lock.unlock();
        print_flag = true;
        break;
      }
      if (print_flag) {
        LOG_WARNING("decoder queue size {} exceed 20, encode suspended.", size);
        print_flag = false;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  encoder.Close();
  LOG_INFO("<=== EncodeImage thread exit. {} images pushed.", iteration);
}

void JpegCodec::DecodeImage(JpegDecoder &decoder, int width, int height,
                            TaskQueue &qin, TaskQueue &qout, Barrier &barrier) {
  int ret = 0;
  int count = 0;

  FrameDevice device = FrameDevice::HDPL;
  std::vector<DecodeData> frm_data(2);

  if (width > RESIZER_MAX_WIDTH || height > RESIZER_MAX_HEIGHT) {
    LOG_ERROR(
        "The decoded outputs exceeds the upper limit supported by resizer.");
    return;
  }
  // create a resizer to process decoded outputs
  auto option =
      tcim::ImageOps::Resizer::Option(tcim::DataFmt::YUV420SP, DEVICE_ID);
  auto md_width = decoder.GetResizer().md_width;
  auto md_height = decoder.GetResizer().md_width;
  int64_t max_width = width < md_width ? md_width : width;
  int64_t max_height = height < md_height ? md_height : height;
  option.SetMaxSize(max_width, max_height);
  if (decoder.SetResizer(option) != VIDEO_DECODER_OK) {
    LOG_ERROR("Create resizer failed.");
    return;
  }

  barrier.barrier();

  while (true) {
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task_info = qin.queue.front();
    if (task_info.is_end) {
      LOG_INFO("===> Decode Image thread EOS received.");
      std::unique_lock<std::mutex> lock_out(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock_out.unlock();
      lock_in.unlock();
      break;
    }
    qin.queue.pop();
    lock_in.unlock();

    auto encoded_buf = task_info.image;
    ret = decoder.PushData((char *)encoded_buf.Data(), encoded_buf.Size());
    if (ret != 0) {
      LOG_ERROR("Push data into JPEG decoder failed: {}.", ret);
      ret = -1;
      break;
    }

    ret = decoder.PullData(frm_data, device);
    if (ret != 0) {
      LOG_ERROR("Pull data from JPEG decoder failed: {}.", ret);
      ret = -1;
      break;
    }

    tcim::Buffer y_buf;
    tcim::Buffer uv_buf;
    tcim::Tensor yuv_tensor;
    tcim::Tensor y_tensor;
    tcim::Tensor uv_tensor;
    auto yuv_info =
        tcim::TensorInfo::CreateYUVInfo(width, height, tcim::YUV420SP);
    std::cout << "-->> decoded yuv info:" << yuv_info << std::endl;
    if (device == FrameDevice::CPU) {
      yuv_tensor = tcim::Tensor::CreateHostTensor(yuv_info);
      y_buf = tcim::Buffer::CreateHostBuffer(frm_data[0].len, frm_data[0].data);
      uv_buf =
          tcim::Buffer::CreateHostBuffer(frm_data[1].len, frm_data[1].data);
    } else {
      yuv_tensor = tcim::Tensor::CreateDeviceTensor(yuv_info);
      y_buf = tcim::Buffer::CreateDeviceBuffer(frm_data[0].data,
                                               frm_data[0].len, DEVICE_ID, "");
      uv_buf = tcim::Buffer::CreateDeviceBuffer(frm_data[1].data,
                                                frm_data[1].len, DEVICE_ID, "");
    }
    yuv_tensor.SplitYUV(y_tensor, uv_tensor);
    y_buf.CopyTo(y_tensor.Buffer());
    uv_buf.CopyTo(uv_tensor.Buffer());

    // resize decoded image from 640x426 -> 1920x1080
    auto img_resizer = decoder.GetResizer();
    auto md_height = img_resizer.md_height;
    auto md_width = img_resizer.md_width;
    tcim::Tensor resized_yuv_tensor;
    auto resized_yuv_info =
        tcim::TensorInfo::CreateYUVInfo(md_width, md_height, tcim::YUV420SP);
    LOG_INFO("-->> resized yuv info: height={}, width={}.", md_height,
             md_width);
    if (device == FrameDevice::CPU) {
      resized_yuv_tensor = tcim::Tensor::CreateHostTensor(resized_yuv_info);
    } else {
      resized_yuv_tensor = tcim::Tensor::CreateDeviceTensor(resized_yuv_info);
    }
    tcim::ImageOps::RectRoi roi(0, 0, width, height);
    tcim::ImageOps::Resizer::RunOption run_opt(roi);
    run_opt.interp_mode = tcim::ImageOps::Resizer::EnInterpMode::Nearest;
    img_resizer.resizer->Run(yuv_tensor, resized_yuv_tensor, run_opt, true);

#ifdef DUMP_JPEG_DECODED_DATA
    auto decoded_tensor = yuv_tensor.ToHost(true);
    auto resized_tensor = resized_yuv_tensor.ToHost(true);

    std::string decoded_file_name = "decoded_" + std::to_string(count);
    SaveImgs(height, width, decoded_tensor.Data(), "debug_results",
             decoded_file_name);
    LOG_INFO("save the decoded image, file_path: {}.", decoded_file_name);

    std::string resized_file_name = "resized_" + std::to_string(count);
    SaveImgs(md_height, md_width, resized_tensor.Data(), "debug_results",
             resized_file_name);
    LOG_INFO("save the resized image, file_path: {}.", resized_file_name);
#endif  // DUMP_JPEG_DECODED_DATA

    decoder.ReleaseBuf();

    // push input data to det queue
    TaskInfo decoded_task_info;
    decoded_task_info.req_id = count++;
    decoded_task_info.image = resized_yuv_tensor.Buffer();

    bool print_flag = true;
    while (1) {
      std::unique_lock<std::mutex> lock(qout.mutex);
      // check if det queue is too full
      int size = qout.queue.size();
      if (size <= DECODER_QUEUE_SIZE) {
        qout.queue.push(decoded_task_info);
        LOG_INFO("Jpeg decoder push data req_id {}, queue size {}.",
                 decoded_task_info.req_id, size);
        qout.cond.notify_all();
        lock.unlock();
        print_flag = true;
        break;
      }
      if (print_flag) {
        LOG_WARNING(
            "detect queue size {} exceed 20, push decoded data suspended.",
            size);
        print_flag = false;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  decoder.Close();
  LOG_INFO("<=== DeocdeImage thread exit. {} jpeg data received.", count);
}
