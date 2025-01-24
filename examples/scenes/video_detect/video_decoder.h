#include <iostream>
#include <sstream>
#include <string>

#include "hm_vpu_hal.h"


#define DECODE_TO_HOST 1  // decoded image copy to host

#define PUSH_LAST_STREAM_DATA_COMPLETE    (0x5a5a)
#define V4L2_PIX_FMT_HEVC   v4l2_fourcc('H', 'E', 'V', 'C')  /* HEVC aka H.265 */

#define VIDEO_DECODER_ERR -1
#define VIDEO_DECODER_OK  0
#define VIDEO_DECODER_EOS 1

typedef struct {
  void* data = nullptr;
  int len = 0;
} DecodeData;


class VideoDecoder {
 public:
  VideoDecoder(const std::string& input_fmt, const std::string& output_fmt) {
    auto in_fmt = GetInputFormat(input_fmt);
    auto out_fmt = GetOutputFormat(output_fmt);
    if (HmCodecDecOpen(0, 0, in_fmt, out_fmt, &vir_fd_)) {
      printf("[error] pid=%d open vpu device fail.\n", getpid());
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
    return ret;
  }

  // Push stream data to decoder
  int PushData(uint8_t *data, int size, int last_flag) {
    return HmCodecDecPushStreamData(vir_fd_, data, size, last_flag);
  }

  // Pull yuv data from decoder
  int PullData(DecodeData& output) {
    int total_len = 0;
    int ret = 0;
    static uint64_t fid = 0;
    static auto begin = GET_TIME();

    ret = HmCodecDecGetAvailBuf(vir_fd_, &buf_info_);
    if (ret != 0) {
      printf("[error] pid=%d get avial buf fail. ret=%d\n", getpid(), ret);
      return VIDEO_DECODER_ERR;
    }

    /* received yuv plane length equal 0 is the flag of end of decoder */
    if (buf_info_.length[0] == 0) {
      printf("EOS received.\n");
      return VIDEO_DECODER_EOS;
    }

    int offset = 0;
#if DECODE_TO_HOST
    /* get yuv data by plane */
    for (int i = 0; i < buf_info_.planes; ++i) {
      // check len
      if (buf_info_.length[i] + offset > output.len) {
        printf("[error] buf[%d] len %d + %d > dst data len %d!\n", i, buf_info_.length[i], offset, output.len);
        return VIDEO_DECODER_ERR;
      }
      ret = HmCodecDecGetYuvPayload(vir_fd_, buf_info_.phy_addr[i], (char*)output.data + offset,
                                    buf_info_.length[i]);
      if (ret != 0) {
        printf("[error] HmCodecDecGetYuvPayload fail!\n");
        return VIDEO_DECODER_ERR;
      }
      offset += buf_info_.length[i];
    }
#else
    for (int i = 0; i < buf_info_.planes; ++i) {
      DecodeData decoded;
      decoded.data = (void*)buf_info_.phy_addr[i];
      decoded.len = buf_info_.length[i];
      data.emplace_back(decoded);
    }
#endif

    ++fid;
    auto cur = GET_TIME();
    auto stamp = GET_COST(begin, cur);
    printf("get frame %ld, stamp %ld, planes %d, len %d\n", fid, stamp, buf_info_.planes, offset);

    return VIDEO_DECODER_OK;
  }

  int ReleaseBuf() {
    int ret = 0;
    /* after using this buf, you need to return it.*/
    ret = HmCodecDecReturnBuf(vir_fd_, buf_info_.index);
    if (ret != 0) {
      printf("pid=%d return buf fail!\n", getpid());
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
      printf("[error] input format %s err!\n", input_fmt.c_str());
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
      printf("[error] output format %s err!\n", output_fmt.c_str());
    }
    return type;
  }

  int32_t input_stream_fps = 30;
  uint64_t vir_fd_ = 0;
  struct planes_info buf_info_;
  int width_ = 0;
  int height_ = 0;
};
