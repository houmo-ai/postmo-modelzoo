#ifndef EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_JPEG_CODEC_H_
#define EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_JPEG_CODEC_H_

#include <condition_variable>
#include <cstring>
#include <iostream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include "hm_jpeg_hal.h"
#include "video_detect_utils.h"

class JpegEncoder {
 public:
  JpegEncoder(const std::string &input_fmt, const std::string &output_fmt,
              const uint32_t &width, const uint32_t &height) {
    auto in_fmt = GetInputFormat(input_fmt);
    auto out_fmt = GetOutputFormat(output_fmt);
    struct jpeg_enc_attr enc_attr;
    enc_attr.input_type = in_fmt;
    enc_attr.output_type = out_fmt;
    enc_attr.width = width;
    enc_attr.height = height;
    if (HmJpegEncOpen(0, 0, &enc_attr, &vir_fd_)) {
      LOG_ERROR("Open vpu device failed!");
    }
    width_ = width;
    height_ = height;
  }

  ~JpegEncoder() {
    if (vir_fd_ != 0) {
      Close();
    }
  }

  int Close() {
    int ret = HmJpegEncClose(vir_fd_);
    vir_fd_ = 0;
    if (host_data_) {
      free(host_data_);
    }
    return ret;
  }

  // Push yuv data to encoder
  int PushData(char *y_buf, char *uv_buf, int last_flag) {
    struct yuv_data data;
    data.yuv_data[0] = y_buf;
    data.yuv_data[1] = uv_buf;
    data.planes = 2;
    data.width = width_;
    data.height = height_;
    data.last = last_flag;
    return HmJpegEncPushYuvData(vir_fd_, &data);
  }

  // Pull yuv data from decoder
  int PullData(EncodedData &enc_data, FrameDevice device) {
    static uint64_t fid = 0;
    static auto begin = GET_TIME();

    int ret = HmJpegEncGetAvailBuf(vir_fd_, &buf_info_);
    if (ret != 0) {
      LOG_ERROR("Get avail buf failed, ret={}", ret);
      return VIDEO_DECODER_ERR;
    }

    if (device == CPU) {
      if (host_data_ == nullptr) {
        host_data_ = malloc(buf_info_.length[0]);
      }
      ret = HmJpegEncGetStreamPayload(vir_fd_, buf_info_.phy_addr[0],
                                      (char *)host_data_, buf_info_.length[0]);
      if (ret != 0) {
        LOG_ERROR("HmJpegEncGetStreamPayload fail!");
        return VIDEO_DECODER_ERR;
      }
      enc_data.data = host_data_;
    } else {
      enc_data.data = (void *)buf_info_.phy_addr[0];
    }
    enc_data.len = buf_info_.length[0];

    ++fid;
    auto cur = GET_TIME();
    auto stamp = GET_COST(begin, cur);
    LOG_INFO("get encoded data {}, stamp {}, planes {}, length {}.", fid, stamp,
             buf_info_.planes, enc_data.len);

    return VIDEO_DECODER_OK;
  }

  int GetWidth() { return width_; }

  int GetHeight() { return height_; }

  int ReleaseBuf() {
    /* after using this buf, you need to return it.*/
    int ret = HmJpegEncReturnBuf(vir_fd_, buf_info_.index);
    if (ret != 0) {
      LOG_ERROR("Return encoder buf failed!");
      return VIDEO_DECODER_ERR;
    }
    return VIDEO_DECODER_OK;
  }

 protected:
  jpeg_encoder_input_fmt_type GetInputFormat(const std::string &input_fmt) {
    jpeg_encoder_input_fmt_type type = NV12_JPEG_ENC;
    if (input_fmt == "NV12M") {
      type = NV12_JPEG_ENC;
    } else if (input_fmt == "NV21M") {
      type = NV21_JPEG_ENC;
    } else if (input_fmt == "YM12") {
      type = YM12_JPEG_ENC;
    } else if (input_fmt == "YM16") {
      type = YM16_JPEG_ENC;
    } else {
      LOG_ERROR("Input format err: {}!", input_fmt);
    }
    return type;
  }

  jpeg_encoder_output_fmt_type GetOutputFormat(const std::string &output_fmt) {
    jpeg_encoder_output_fmt_type type = JPEG_ENC;
    if (output_fmt == "JPEG") {
      type = JPEG_ENC;
    } else {
      LOG_ERROR("Output format err: {}!", output_fmt);
    }
    return type;
  }

  uint64_t vir_fd_ = 0;
  struct planes_info buf_info_;
  void *host_data_ = nullptr;
  int width_ = 0;
  int height_ = 0;
};

class JpegDecoder {
 public:
  JpegDecoder(const std::string &input_fmt, const std::string &output_fmt) {
    auto in_fmt = GetInputFormat(input_fmt);
    auto out_fmt = GetOutputFormat(output_fmt);
    if (HmJpegDecOpen(0, 0, in_fmt, out_fmt, &vir_fd_)) {
      LOG_ERROR("Open vpu device failed!");
    }
    img_resizer_.resizer = new tcim::ImageOps::Resizer();
  }

  ~JpegDecoder() {
    if (vir_fd_ != 0) {
      Close();
    }
  }

  int Close() {
    int ret = HmJpegDecClose(vir_fd_);
    vir_fd_ = 0;
    for (int i = 0; i < 2; i++) {
      if (host_data_[i]) {
        free(host_data_[i]);
      }
    }
    delete img_resizer_.resizer;
    return ret;
  }

  // Push stream data to decoder
  int PushData(char *data, int size) {
    LOG_INFO("Jpeg decoder {} push data, size: {}.", vir_fd_, size);
    return HmJpegDecPushStream(vir_fd_, data, size);
  }

  // Pull yuv data from decoder
  int PullData(std::vector<DecodeData> &frm_data, FrameDevice device) {
    LOG_INFO("Jpeg decoder {} pull data.", vir_fd_);
    static uint64_t fid = 0;
    static auto begin = GET_TIME();

    int ret = HmJpegDecGetAvailBuf(vir_fd_, &buf_info_);
    if (ret != 0) {
      LOG_ERROR("Get avail buf failed, ret={}", ret);
      return VIDEO_DECODER_ERR;
    }

    int total_length = 0;
    /* get yuv data by plane */
    for (int i = 0; i < buf_info_.planes; ++i) {
      if (device == FrameDevice::CPU) {
        if (host_data_[i] == nullptr) {
          host_data_[i] = malloc(buf_info_.length[i]);
        }
        ret =
            HmJpegDecGetYuvPayload(vir_fd_, buf_info_.phy_addr[i],
                                   (char *)host_data_[i], buf_info_.length[i]);
        if (ret != 0) {
          LOG_ERROR("HmJpegDecGetYuvPayload fail!");
          return VIDEO_DECODER_ERR;
        }
        frm_data[i].data = host_data_[i];
      } else {
        frm_data[i].data = (void *)buf_info_.phy_addr[i];
      }
      frm_data[i].len = buf_info_.length[i];
      total_length += buf_info_.length[i];
    }

    ++fid;
    auto cur = GET_TIME();
    auto stamp = GET_COST(begin, cur);
    LOG_INFO("get frame {}, stamp {}, planes {}, total length {}.", fid, stamp,
             buf_info_.planes, total_length);

    return VIDEO_DECODER_OK;
  }

  int ReleaseBuf() {
    int ret = 0;
    /* after using this buf, you need to return it.*/
    ret = HmJpegDecReturnBuf(vir_fd_, buf_info_.index);
    if (ret != 0) {
      LOG_ERROR("Return Jpeg decoder buf failed!");
      return VIDEO_DECODER_ERR;
    }
    return VIDEO_DECODER_OK;
  }

  int SetModelInfo(const int64_t &width, const int64_t &heigth) {
    img_resizer_.md_width = width;
    img_resizer_.md_height = heigth;

    return VIDEO_DECODER_OK;
  }

  int SetResizer(tcim::ImageOps::Resizer::Option &option) {
    int ret = 0;
    ret = img_resizer_.resizer->Init(option);
    if (ret != 0) {
      LOG_ERROR("Init resizer failed!");
      return VIDEO_DECODER_ERR;
    }
    img_resizer_.max_width = option.max_w;
    img_resizer_.max_height = option.max_h;

    return VIDEO_DECODER_OK;
  }

  ImgResizer &GetResizer() { return img_resizer_; }

 protected:
  jpeg_decoder_input_fmt_type GetInputFormat(const std::string &input_fmt) {
    jpeg_decoder_input_fmt_type type = JPEG;
    if (input_fmt == "JPEG") {
      type = JPEG;
    } else {
      LOG_ERROR("Input format err: {}!", input_fmt);
    }
    return type;
  }

  jpeg_decoder_output_fmt_type GetOutputFormat(const std::string &output_fmt) {
    jpeg_decoder_output_fmt_type type = NV12_JPEG;
    if (output_fmt == "NV12M") {
      type = NV12_JPEG;
    } else if (output_fmt == "NV21M") {
      type = NV21_JPEG;
    } else if (output_fmt == "YM12") {
      type = YM12_JPEG;
    } else if (output_fmt == "YM16") {
      type = YM16_JPEG;
    } else {
      LOG_ERROR("Output format err: {}!", output_fmt);
    }
    return type;
  }

  uint64_t vir_fd_ = 0;
  struct planes_info buf_info_;
  ImgResizer img_resizer_;
  void *host_data_[2] = {nullptr};
  int width_ = 0;
  int height_ = 0;
};

class JpegCodec {
 public:
  void EncodeImage(JpegEncoder &encoder, char *yuv_buffer, int iteration,
                   TaskQueue &qout, Barrier &barrier);
  void DecodeImage(JpegDecoder &decoder, int width, int height, TaskQueue &qin,
                   TaskQueue &qout, Barrier &barrier);
};

#endif  // EXAMPLES_SCENES_VIDEO_DETECT_INCLUDE_JPEG_CODEC_H_