#ifndef EXAMPLES_SCENES_VIDEO_DETECT_H264_H_
#define EXAMPLES_SCENES_VIDEO_DETECT_H264_H_

extern "C" {
#include <libavformat/avformat.h>
}

#include <string>
#include <vector>

class H264Node {
 public:
  static H264Node *CreateH264Node(const std::string &h264_file_path);
  ~H264Node();
  bool HasNextFrame();
  bool GetNextFrame(uint8_t **data, int32_t *size);

 private:
  explicit H264Node(const std::string &h264_file_path);
  AVFormatContext *_fmt_ctx = nullptr;
  int32_t _video_stream_idx = -1;
  AVPacket *_packet = nullptr;
  bool _need_unref_packet = false;
};

class H264DataSource {
 public:
  static H264DataSource *CreateH264Source(std::string h264_path,
                                          size_t iterations);
  ~H264DataSource();
  bool GetData(uint8_t **data, int32_t *size);

 private:
  explicit H264DataSource(size_t iterations);
  std::vector<uint8_t *> _dataptr_list;
  std::vector<size_t> _data_size_list;
  size_t _data_num;
  size_t _iterations;
  size_t _loop_idx;
};
#endif  // EXAMPLES_SCENES_VIDEO_DETECT_H264_H_
