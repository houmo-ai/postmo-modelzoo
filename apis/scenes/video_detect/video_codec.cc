#include <getopt.h>
#include <linux/videodev2.h>
#include <stdio.h>
#include <unistd.h>

#include <chrono>
#include <condition_variable>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

extern "C" {
#include "libavcodec/avcodec.h"
#include "libavformat/avformat.h"
}

#include "video_codec.h"

#ifdef RK_DECODER
#include "h264.h"
#include "mpp_frame.h"
#include "rk_mpi.h"
#include "rk_type.h"
// #define DUMP_RK_DECODED_DATA  // save rk decoded results
#endif

// #define DUMP_HM_DECODED_DATA  // save hm decoded results

std::atomic<int32_t> g_codec_cnt = 0;

#ifdef RK_DECODER
void VideoCodec::PushRKStream(PushStreamInfo &stream_info, TaskQueue &qout,
                              Barrier &barrier) {
  // when iteration is greater than 1, reusing the decoded data
  int iteration = 1;
  H264DataSource *data_source =
      H264DataSource::CreateH264Source(stream_info.stream_path, iteration);
  MppCtx ctx = NULL;
  MppApi *mpi = NULL;
  MppBuffer frm_buf = NULL;
  MppDecCfg cfg = NULL;
  MPP_RET ret = MPP_OK;
  MppCodingType type = MppCodingType::MPP_VIDEO_CodingAVC;

  ret = mpp_create(&ctx, &mpi);
  if (ret) {
    LOG_ERROR("[RK Decoder] mpp_create failed!");
    return;
  }
  ret = mpp_init(ctx, MPP_CTX_DEC, type);
  if (ret) {
    LOG_ERROR("[RK Decoder] mpp_init failed!");
    return;
  }
  mpp_dec_cfg_init(&cfg);
  // get default config from decoder context
  ret = mpi->control(ctx, MPP_DEC_GET_CFG, cfg);
  if (ret) {
    LOG_ERROR("[RK Decoder] mpp ctx {} failed to get decoder cfg, ret {}.", ctx,
              (int)ret);
    return;
  }

  barrier.barrier();
  LOG_INFO("[RK Decoder] {}, mpp ctx {} start to decode stream...",
           reinterpret_cast<void *>(this), ctx);

  uint64_t fid = 0;
  while (true) {
    uint8_t *data = nullptr;
    MppPacket packet = NULL;
    int32_t data_length = 0;
    ret = mpp_packet_init(&packet, NULL, 0);
    if (ret) {
      LOG_ERROR("[RK Decoder] mpp_packet_init failed!");
      break;
    }
    if (!data_source->GetData(&data, &data_length)) {
      LOG_WARNING("[RK Decoder] No data in data source at all for.");
      break;
    }

    mpp_packet_set_data(packet, data);
    mpp_packet_set_size(packet, data_length);
    mpp_packet_set_pos(packet, data);
    mpp_packet_set_length(packet, data_length);
    ret = mpi->decode_put_packet(ctx, packet);
    if (MPP_OK != ret) {
      LOG_ERROR("[RK Decoder] mpp ctx {} decode_put_packet failed, ret {}", ctx,
                (int)ret);
      continue;
    }

    while (true) {
      MppFrame frame = NULL;
      RK_S32 times = 30;
      do {
        ret = mpi->decode_get_frame(ctx, &frame);
        if (MPP_ERR_TIMEOUT == ret) {
          times--;
        } else {
          break;
        }
      } while (times > 0);
      if (frame) {
        this->decoded_cnt++;
        if (mpp_frame_get_info_change(frame)) {
          RK_U32 width = mpp_frame_get_width(frame);
          RK_U32 height = mpp_frame_get_height(frame);
          RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
          RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);
          RK_U32 buf_size = mpp_frame_get_buf_size(frame);
          LOG_INFO(
              "[RK Decoder] mpp ctx {} frame info change: width={}, height={}, "
              "hor_stride={}, ver_stride={}, buf_size={}",
              ctx, width, height, hor_stride, ver_stride, buf_size);
          ret = mpi->control(ctx, MPP_DEC_SET_INFO_CHANGE_READY, NULL);
          if (ret) {
            LOG_ERROR(
                "[RK Decoder] mpp ctx {} info change ready failed, ret {}", ctx,
                (int)ret);
          }
        } else {
          // drop Frames
          // if (fid % 6 != 0) {
          //   fid++;
          //   mpp_frame_deinit(&frame);
          //   continue;
          // }
          MppBuffer buffer = NULL;
          RK_U8 *base = NULL;
          buffer = mpp_frame_get_buffer(frame);
          if (NULL == buffer) {
            break;
          }

          RK_U32 width = mpp_frame_get_width(frame);
          RK_U32 height = mpp_frame_get_height(frame);
          RK_U32 hor_stride = mpp_frame_get_hor_stride(frame);
          RK_U32 ver_stride = mpp_frame_get_ver_stride(frame);
          base = (RK_U8 *)mpp_buffer_get_ptr(buffer);
          RK_U32 buf_size = mpp_frame_get_buf_size(frame);

          // copy the decoded result to a contiguous buffer
          RK_U32 i;
          RK_U8 *base_y = base;
          RK_U8 *base_c = base + hor_stride * ver_stride;
          size_t frame_size = (height * 3 / 2) * width;
          auto result = std::shared_ptr<void>(malloc(frame_size), free);
          memcpy(result.get(), base_y, (width * height));
          memcpy(result.get() + (width * height), base_c, (width * height / 2));

          // construct TaskInfo for inference
          TaskInfo task_info;
          task_info.req_id = fid++;
          task_info.buffer = result;
          task_info.buffer_length = frame_size;

          bool print_flag = true;
          while (1) {
            std::unique_lock<std::mutex> lock(qout.mutex);
            // check if det queue is too full
            int size = qout.queue.size();
            if (size <= DECODER_QUEUE_SIZE) {
              qout.queue.push(task_info);
              qout.cond.notify_all();
              lock.unlock();
              break;
            }
            if (print_flag) {
              LOG_WARNING(
                  "[RK Decoder] push rk stream queue size {} exceed 20, push "
                  "stream suspended.",
                  size);
              print_flag = false;
            }
            lock.unlock();
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
          }
        }
        mpp_frame_deinit(&frame);
      } else {
        break;
      }
    }
    if (packet) {
      mpp_packet_deinit(&packet);
      packet = NULL;
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(33));
  }

  // construct eos TaskInfo
  TaskInfo task_info;
  task_info.req_id = fid++;
  task_info.is_end = true;
  {
    std::unique_lock<std::mutex> lock(qout.mutex);
    qout.queue.push(task_info);
    LOG_INFO("===> [RK Decoder] push eos task, req_id {}.", task_info.req_id);
    qout.cond.notify_all();
    lock.unlock();
  }

  ret = mpi->reset(ctx);
  if (ret) {
    LOG_ERROR("[RK Decoder] mpp ctx {} reset failed, ret {}.", ctx, (int)ret);
    return;
  }
  if (ctx) {
    mpp_destroy(ctx);
    ctx = NULL;
  }
  delete data_source;

  LOG_INFO("<=== PushRKStream thread exit. {} frames received.", --fid);
}
#endif

void VideoCodec::PushStream(std::shared_ptr<VideoDecoder> decoder,
                            PushStreamInfo &stream_info, Barrier &barrier) {
  AVFormatContext *format_ctx = NULL;
  AVPacket *packet = NULL;
  int ret = 0;
  uint32_t push_slices_cnt = 0;
  uint32_t rcv_I_slice_flag = 0;
  int count = 0;

  /*
   * there maybe many streams in a media file (video stream or other audio
   * streams). check out check video stream
   */
  int video_stream_index = -1;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
  av_register_all();
#endif
  avformat_network_init();

  // av_log_set_level(AV_LOG_DEBUG);

  /* open stream */
  ret = avformat_open_input(&format_ctx, stream_info.stream_path.c_str(), NULL,
                            NULL);
  if (ret < 0) {
    char error_buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, error_buf, sizeof(error_buf));
    LOG_ERROR("Open stream {} failed, error msg: {}.", stream_info.stream_path,
              error_buf);
    return;
  }
  LOG_INFO("open stream {} completed.", stream_info.stream_path);

  ret = avformat_find_stream_info(format_ctx, NULL);
  if (ret < 0) {
    char error_buf[AV_ERROR_MAX_STRING_SIZE];
    av_strerror(ret, error_buf, sizeof(error_buf));
    LOG_ERROR("Find stream info failed, error msg: {}.", error_buf);
    return;
  }

  for (int i = 0; i < format_ctx->nb_streams; i++) {
    if (format_ctx->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
      video_stream_index = i;
      break;
    }
  }

  if (video_stream_index == -1) {
    LOG_ERROR("Can't find video stream!");
    avformat_close_input(&format_ctx);
    return;
  }

  AVCodecParameters *codecParameters =
      format_ctx->streams[video_stream_index]->codecpar;
  LOG_INFO("frame size is {} x {}.", codecParameters->width,
           codecParameters->height);
  auto &width = decoder->GetWidth();
  width = codecParameters->width;
  auto &height = decoder->GetHeight();
  height = codecParameters->height;

#ifdef RESIZER
  if (width > RESIZER_MAX_WIDTH || height > RESIZER_MAX_HEIGHT) {
    LOG_ERROR(
        "The decoded outputs exceeds the upper limit supported by resizer.");
    avformat_close_input(&format_ctx);
    return;
  }
  // create a resizer to process decoded outputs
  auto option =
      tcim::ImageOps::Resizer::Option(tcim::DataFmt::YUV420SP, DEVICE_ID);
  auto md_width = decoder->GetResizer().md_width;
  auto md_height = decoder->GetResizer().md_width;
  int64_t max_width = width < md_width ? md_width : width;
  int64_t max_height = height < md_height ? md_height : height;
  option.SetMaxSize(max_width, max_height);
  if (decoder->SetResizer(option) != VIDEO_DECODER_OK) {
    LOG_ERROR("Create resizer failed.");
    avformat_close_input(&format_ctx);
    return;
  }
#endif

  barrier.barrier();

  packet = av_packet_alloc();
  if (packet == NULL) {
    LOG_ERROR("Alloc packet failed");
    avformat_close_input(&format_ctx);
    return;
  }

  while (1) {
    if (count >= stream_info.frame_limit && stream_info.frame_limit != 0) {
      LOG_WARNING("===> push stream frame limit reached: {}.",
                  stream_info.frame_limit);
      /* push last flag */
      ret = decoder->PushData(nullptr, 0, 1);
      if (ret != 0) {
        LOG_ERROR("push last stream failed, ret: {}", ret);
        ret = -1;
      }
      break;
    }
    ret = av_read_frame(format_ctx, packet);
    if (ret < 0) {
      if (packet->size == 0) {
        LOG_INFO("===> push stream thread EOS received");
        /* push last flag */
        ret = decoder->PushData(packet->data, packet->size, 1);
        if (ret != 0) {
          LOG_ERROR("Push last stream failed, ret: {}.", ret);
          ret = -1;
        }
      } else {
        char error_buf[AV_ERROR_MAX_STRING_SIZE];
        av_strerror(ret, error_buf, sizeof(error_buf));
        LOG_ERROR("av_read_frame fail, error msg: {}.", error_buf);
        ret = -1;
      }
      /* exit loop */
      break;
    }

    /* wait I slice, when decoder start work. */
    if (packet->flags & AV_PKT_FLAG_KEY) rcv_I_slice_flag = 1;

    if (rcv_I_slice_flag != 1) {
      LOG_DEBUG("skip non-key frame");
      av_packet_unref(packet);
      continue;
    }

    if (packet->stream_index == video_stream_index) {
      /* find which buf is available. if find, so push stream */
      push_slices_cnt++;

      ret = decoder->PushData(packet->data, packet->size, 0);
      if (ret != 0) {
        LOG_ERROR("push stream failed: {}.", ret);
        ret = -1;
        break;
      }

      count++;
    }
    av_packet_unref(packet);
    std::this_thread::sleep_for(std::chrono::milliseconds(33));
  }

  av_packet_free(&packet);
  avformat_close_input(&format_ctx);

  LOG_INFO("<=== PushStream thread exit. {} frames received.", count);
}

void VideoCodec::GetFrame(std::shared_ptr<VideoDecoder> decoder,
                          TaskQueue &qout, Barrier &barrier) {
  int fid = 0;
  int ret = 0;
  int count = 0;

  FrameDevice device = HDPL;
  std::vector<DecodeData> frm_data(2);

  barrier.barrier();

  int height = decoder->GetHeight();
  int width = decoder->GetWidth();
  LOG_INFO("===> decoder heigth: {}, width: {}.", height, width);

  while (1) {
    ret = decoder->PullData(frm_data, device);
    this->decoded_cnt++;
    if (ret == VIDEO_DECODER_EOS) {
      TaskInfo task_info;
      task_info.is_end = true;
      std::unique_lock<std::mutex> lock(qout.mutex);
      qout.queue.push(task_info);
      qout.cond.notify_all();
      lock.unlock();
      LOG_INFO("===> get frame thread EOS received.");
      break;
    }
    if (ret != VIDEO_DECODER_OK) {
      LOG_ERROR("decoder pull data fail: {}", ret);
      continue;
    }

    tcim::Buffer y_buf;
    tcim::Buffer uv_buf;
    tcim::Tensor yuv_tensor;
    tcim::Tensor y_tensor;
    tcim::Tensor uv_tensor;
    auto yuv_info =
        tcim::TensorInfo::CreateYUVInfo(width, height, tcim::YUV420SP);
    std::cout << "-->> decoded yuv info:" << yuv_info << std::endl;
    if (device == CPU) {
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

#ifdef RESIZER
    auto img_resizer = decoder->GetResizer();
    auto md_heigth = img_resizer.md_height;
    auto md_width = img_resizer.md_width;
    tcim::Tensor resized_yuv_tensor;
    auto resized_yuv_info =
        tcim::TensorInfo::CreateYUVInfo(md_width, md_heigth, tcim::YUV420SP);
    std::cout << "-->> resized yuv info:" << resized_yuv_info << std::endl;
    ;
    if (device == CPU) {
      resized_yuv_tensor = tcim::Tensor::CreateHostTensor(resized_yuv_info);
    } else {
      resized_yuv_tensor = tcim::Tensor::CreateDeviceTensor(resized_yuv_info);
    }
    tcim::ImageOps::RectRoi roi(0, 0, width, height);
    tcim::ImageOps::Resizer::RunOption run_opt(roi);
    run_opt.interp_mode = tcim::ImageOps::Resizer::EnInterpMode::Nearest;
    img_resizer.resizer->Run(yuv_tensor, resized_yuv_tensor, run_opt, true);

#ifdef DUMP_HM_DECODED_DATA
    auto decoded_tensor = yuv_tensor.ToHost(true);
    auto resized_tensor = resized_yuv_tensor.ToHost(true);

    std::string decoded_file_name = "decoded_" + std::to_string(fid);
    SaveImgs(height, width, decoded_tensor.Data(), "debug_results",
             decoded_file_name);
    LOG_INFO("save the decoded image, file_path: {}.", decoded_file_name);

    std::string resized_file_name = "resized_" + std::to_string(fid);
    SaveImgs(md_heigth, md_width, resized_tensor.Data(), "debug_results",
             resized_file_name);
    LOG_INFO("save the resized image, file_path: {}.", resized_file_name);
#endif  // DUMP_HM_DECODED_DATA
#endif  // RESIZER

    count++;
    decoder->ReleaseBuf();

    // push input data to det queue
    TaskInfo task_info;
    task_info.req_id = fid++;
#ifdef RESIZER
    task_info.image = resized_yuv_tensor.Buffer();
#else
    task_info.image = yuv_tensor.Buffer();
#endif  // RESIZER

    bool print_flag = true;
    while (1) {
      std::unique_lock<std::mutex> lock(qout.mutex);
      // check if det queue is too full
      int size = qout.queue.size();
      if (size <= DECODER_QUEUE_SIZE) {
        qout.queue.push(task_info);
        LOG_INFO("decoder pull data req_id {}, queue size {}.",
                 task_info.req_id, size);
        qout.cond.notify_all();
        lock.unlock();
        print_flag = true;
        break;
      }
      if (print_flag) {
        LOG_WARNING("detect queue size {} exceed 20, get frame suspended.",
                    size);
        print_flag = false;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
  }

  decoder->Close();
  LOG_INFO("<=== GetFrame thread exit. {} frames received.", count);
}

void VideoCodec::PushEncodeStream(std::shared_ptr<VideoEncoder> encoder,
                                  TaskQueue &qin, Barrier &barrier) {
  int ret = 0;
  int frames = 0;
  auto width = encoder->GetWidth();
  auto height = encoder->GetHeight();
  const int y_size = width * height;
  LOG_INFO("===> PushEncodeStream thread start, size {}x{}, ysize={}.", width,
           height, y_size);

  barrier.barrier();

  while (true) {
    // get data from the task queue
    std::unique_lock<std::mutex> lock_in(qin.mutex);
    while (qin.queue.empty()) {
      qin.cond.wait(lock_in);
    }
    auto task_info = qin.queue.front();
    qin.queue.pop();
    lock_in.unlock();

    int last_flag = task_info.is_end == true ? 1 : 0;
    if (last_flag == 1) {
      LOG_INFO("===> Push EOS data into video encoder.");
      ret = encoder->PushData(nullptr, nullptr, last_flag);
      break;
    }

    frames++;
    void *yuv_buffer = task_info.image.Data();
    char *y_buf = (char *)yuv_buffer;
    char *uv_buf = (char *)yuv_buffer + y_size;
    ret = encoder->PushData(y_buf, uv_buf, last_flag);
    if (ret != 0) {
      LOG_ERROR("Push data into video encoder failed: {}.", ret);
      continue;
    }
  }

  LOG_INFO("<=== PushEncodeStream thread exit. {} images pushed.", frames);
}

void VideoCodec::GetEncodeStream(std::shared_ptr<VideoEncoder> encoder,
                                 const int32_t &width, const int32_t &height,
                                 const std::string &output_path,
                                 const int32_t &codec_num, bool &stop_flag,
                                 Barrier &barrier) {
  int ret = 0;
  FrameDevice device = CPU;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
  av_register_all();
#endif
  avformat_network_init();

  AVFormatContext *ofmt_ctx = nullptr;
  ret = avformat_alloc_output_context2(&ofmt_ctx, nullptr, nullptr,
                                       output_path.c_str());
  if (ret < 0) {
    LOG_ERROR("Failed to create ffmpeg output context, error is {}.", ret);
    return;
  }

  AVStream *out_stream = avformat_new_stream(ofmt_ctx, nullptr);
  if (!out_stream) {
    LOG_ERROR("Failed to create output stream.");
    avformat_free_context(ofmt_ctx);
    return;
  }

  int fps_num = 30;
  AVRational fps = {fps_num, 1};
  AVCodecParameters *codecpar = out_stream->codecpar;
  codecpar->codec_id = AV_CODEC_ID_H264;
  codecpar->codec_type = AVMEDIA_TYPE_VIDEO;
  codecpar->width = width;
  codecpar->height = height;
  codecpar->format = AV_PIX_FMT_YUV420P;
  // 设置时间基（1/fps）和帧率
  out_stream->time_base = {1, fps_num};
  out_stream->avg_frame_rate = fps;

  if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    ret = avio_open(&ofmt_ctx->pb, output_path.c_str(), AVIO_FLAG_WRITE);
    if (ret < 0) {
      LOG_ERROR("Failed to open output file, error is {}.", ret);
      avformat_free_context(ofmt_ctx);
      return;
    }
  }

  ret = avformat_write_header(ofmt_ctx, nullptr);
  if (ret < 0) {
    LOG_ERROR("Failed to write video header, error is {}.", ret);
    avio_closep(&ofmt_ctx->pb);
    avformat_free_context(ofmt_ctx);
    return;
  }
  int64_t pts = 0;

  LOG_INFO(
      "===> GetEncodeStream thread start, ffmpeg ctx {}, frame size is {}x{}, "
      "fps is {}.",
      reinterpret_cast<void *>(ofmt_ctx), width, height, fps_num);

  barrier.barrier();

  while (true) {
    EncodedData h264_data = {0};
    ret = encoder->PullData(h264_data, device);
    this->encoded_cnt++;
    if (ret == VIDEO_DECODER_EOS) {
      LOG_INFO("===> get encoded frame thread EOS received.");
      encoder->ReleaseBuf();
      break;
    }
    if (ret != VIDEO_DECODER_OK) {
      LOG_ERROR("Pull data from Video encoder failed: {}.", ret);
      continue;
    }

    // 创建AVPacket
    AVPacket *pkt = av_packet_alloc();
    av_init_packet(pkt);
    pkt->data = reinterpret_cast<uint8_t *>(h264_data.data);  // H.264数据指针
    pkt->size = h264_data.len;                                // 数据大小
    pkt->stream_index = out_stream->index;                    // 流索引
    pkt->pts = pts;
    pkt->dts = pts;
    pkt->duration = 1;  // 每帧持续1个时间单位（如1/30秒）
    pts += pkt->duration;
    LOG_DEBUG("Write packet {}, ptr: {}, length: {}.", pts,
              reinterpret_cast<void *>(pkt->data), pkt->size);
    ret = av_interleaved_write_frame(ofmt_ctx, pkt);
    if (ret < 0) {
      LOG_ERROR("Write packet {} failed: {}.", pts, ret);
    }
    av_packet_free(&pkt);  // 释放数据包

    encoder->ReleaseBuf();
  }

  av_write_trailer(ofmt_ctx);
  if (!(ofmt_ctx->oformat->flags & AVFMT_NOFILE)) {
    avio_closep(&ofmt_ctx->pb);
  }
  avformat_free_context(ofmt_ctx);

  encoder->Close();

  g_codec_cnt++;
  if (g_codec_cnt >= codec_num) {
    LOG_INFO("All task done, stop stats thread.");
    stop_flag = true;
  }
  LOG_INFO("<=== GetEncodeStream thread exit. {} images encoded.",
           this->encoded_cnt);
}