#include <iostream>
#include <sstream>
#include <string>

#include "hm_vpu_hal.h"
#include "log.hpp"
#include "utils.hpp"

#define PUSH_LAST_STREAM_DATA_COMPLETE    (0x5a5a)
#define V4L2_PIX_FMT_HEVC   v4l2_fourcc('H', 'E', 'V', 'C')  /* HEVC aka H.265 */

#define VIDEO_DECODER_ERR -1
#define VIDEO_DECODER_OK  0
#define VIDEO_DECODER_EOS 1

typedef enum {
  CPU = 0,
  HDPL = 1,
} FrameDevice;

typedef struct {
  void* data = nullptr;
  size_t len = 0;
} DecodeData;


class VideoDecoder {
 public:
  VideoDecoder(const std::string& input_fmt, const std::string& output_fmt) {
    auto in_fmt = GetInputFormat(input_fmt);
    auto out_fmt = GetOutputFormat(output_fmt);
    if (HmCodecDecOpen(0, 0, in_fmt, out_fmt, &vir_fd_)) {
      LOG_ERROR << "open vpu device fail. pid=" << getpid();
    }
  }

  ~VideoDecoder() {
    if (vir_fd_ != 0) {
      Close();
    }
  }

  int Close() {
    int ret = HmCodecDecClose(vir_fd_);
    vir_fd_ = 0;
    for (int i = 0; i < 2; i++) {
      if (host_data_[i]) {
        free(host_data_[i]);
      }
    }
    return ret;
  }

  // Push stream data to decoder
  int PushData(uint8_t *data, int size, int last_flag) {
    return HmCodecDecPushStreamData(vir_fd_, data, size, last_flag);
  }

  // Pull yuv data from decoder
  int PullData(std::vector<DecodeData>& frm_data, FrameDevice device) {
    int total_len = 0;
    int ret = 0;
    static uint64_t fid = 0;
    static auto begin = GET_TIME();

    ret = HmCodecDecGetAvailBuf(vir_fd_, &buf_info_);
    if (ret != 0) {
      LOG_ERROR << "get avail buf fail. pid=" << getpid() << " ret=" << ret;
      return VIDEO_DECODER_ERR;
    }

    /* received yuv plane length equal 0 is the flag of end of decoder */
    if (buf_info_.length[0] == 0) {
      LOG_DEBUG << "EOS received.";
      return VIDEO_DECODER_EOS;
    }

    int offset = 0;

    /* get yuv data by plane */
    for (int i = 0; i < buf_info_.planes; ++i) {
      if (device == CPU) {
        if (host_data_[i] == nullptr) {
          host_data_[i] = malloc(buf_info_.length[i]);
        }
        ret = HmCodecDecGetYuvPayload(vir_fd_,
                                      buf_info_.phy_addr[i],
                                      (char*)host_data_[i],
                                      buf_info_.length[i]);
        if (ret != 0) {
          LOG_ERROR << "HmCodecDecGetYuvPayload fail!";
          return VIDEO_DECODER_ERR;
        }
        frm_data[i].data = host_data_[i];
        frm_data[i].len = buf_info_.length[i];
      } else {
        frm_data[i].data = (void*)buf_info_.phy_addr[i];
        frm_data[i].len = buf_info_.length[i];
      }
    }

    ++fid;
    auto cur = GET_TIME();
    auto stamp = GET_COST(begin, cur);
    LOG_DEBUG << "get frame " << fid << ", stamp " << stamp << ", planes "
              << buf_info_.planes << ", len " << offset;

    return VIDEO_DECODER_OK;
  }

  int ReleaseBuf() {
    int ret = 0;
    /* after using this buf, you need to return it.*/
    ret = HmCodecDecReturnBuf(vir_fd_, buf_info_.index);
    if (ret != 0) {
      LOG_ERROR << "return buf fail! pid=" << getpid();
      return VIDEO_DECODER_ERR;
    }
    return VIDEO_DECODER_OK;
  }

  int& GetWidth() {
    return width_;
  }

  int& GetHeight() {
    return height_;
  }

 protected:
  decoder_input_fmt_type GetInputFormat(const std::string& input_fmt) {
    decoder_input_fmt_type type = H264;
    if (input_fmt == "H264") {
      type = H264;
    } else if (input_fmt == "H265") {
      type = H265;
    } else {
      LOG_ERROR << "input format err: " << input_fmt;
    }
    return type;
  }

  decoder_output_fmt_type GetOutputFormat(const std::string& output_fmt) {
    decoder_output_fmt_type type = NV12M;
    if (output_fmt == "NV12M") {
      type = NV12M;
    } else if (output_fmt == "NV21M") {
      type = NV21M;
    } else if (output_fmt == "YM12") {
      type = YM12;
    } else if (output_fmt == "YM16") {
      type = YM16;
    } else {
      LOG_ERROR << "output format err : " << output_fmt;
    }
    return type;
  }

  int32_t input_stream_fps = 30;
  uint64_t vir_fd_ = 0;
  struct planes_info buf_info_;
  void* host_data_[2] = {nullptr};
  int width_ = 0;
  int height_ = 0;
};
