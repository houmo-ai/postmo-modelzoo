#include "h264.h"
#include "log.hpp"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/frame.h>
}

#include <fstream>
#include <iostream>
#include <memory>

H264Node::H264Node(const std::string &mp4_file_path) {
  if (avformat_open_input(&(this->_fmt_ctx), mp4_file_path.c_str(), nullptr,
                          nullptr) != 0) {
    LOG_ERROR << "Open " << mp4_file_path << " failed.";
    return;
  }
  if (avformat_find_stream_info(this->_fmt_ctx, nullptr) < 0) {
    LOG_ERROR << "Find stream failed.";
    return;
  }
  for (unsigned int i = 0; i < this->_fmt_ctx->nb_streams; i++) {
    if (this->_fmt_ctx->streams[i]->codecpar->codec_type ==
        AVMEDIA_TYPE_VIDEO) {
      this->_video_stream_idx = i;
      break;
    }
  }
  if (this->_video_stream_idx == -1) {
    LOG_ERROR << "No video stream find in " << mp4_file_path;
  }
  this->_packet = av_packet_alloc();
}

H264Node *H264Node::CreateH264Node(const std::string &mp4_file_path) {
  std::ifstream f(mp4_file_path.c_str());
  if (!f.good()) {
    LOG_ERROR << mp4_file_path << " is not exist.";
    return nullptr;
  }
  return new H264Node(mp4_file_path);
}

H264Node::~H264Node() {
  av_packet_free(&this->_packet);
  avformat_close_input(&(this->_fmt_ctx));
  avformat_free_context(this->_fmt_ctx);
}

bool H264Node::HasNextFrame() {
  if (this->_need_unref_packet == true) {
    av_packet_unref(this->_packet);
    this->_need_unref_packet = false;
  }
  while (true) {
    if (av_read_frame(this->_fmt_ctx, this->_packet) >= 0) {
      this->_need_unref_packet = true;
      if (this->_packet->stream_index != this->_video_stream_idx) {
        av_packet_unref(this->_packet);
        this->_need_unref_packet = false;
        continue;
      }
      return true;
    } else {
      return false;
    }
  }
  return false;
}

bool H264Node::GetNextFrame(uint8_t **data, int32_t *size) {
  *data = this->_packet->data;
  *size = this->_packet->size;
  return true;
}

H264DataSource *H264DataSource::CreateH264Source(std::string h264_path,
                                                 size_t iterations) {
  std::shared_ptr<H264Node> source_node(H264Node::CreateH264Node(h264_path));
  H264DataSource *data_source = new H264DataSource(iterations);
  while (source_node->HasNextFrame()) {
    uint8_t *data_ptr = nullptr;
    int32_t data_size = -1;
    if (source_node->GetNextFrame(&data_ptr, &data_size)) {
      uint8_t *local_data_ptr =
          reinterpret_cast<uint8_t *>(malloc(sizeof(uint8_t) * data_size));
      memcpy(local_data_ptr, data_ptr, sizeof(uint8_t) * data_size);
      data_source->_dataptr_list.push_back(local_data_ptr);
      data_source->_data_size_list.push_back(data_size);
    }
  }
  data_source->_data_num = iterations * data_source->_dataptr_list.size();
  // LOG_INFO << "After CreateH264Source:" << data_source->_data_num;
  return data_source;
}

H264DataSource::H264DataSource(size_t iterations)
    : _iterations(iterations), _loop_idx(0) {}

H264DataSource::~H264DataSource() {
  for (auto data_ptr : this->_dataptr_list) {
    free(data_ptr);
  }
  this->_dataptr_list.clear();
  this->_data_size_list.clear();
}

bool H264DataSource::GetData(uint8_t **data, int32_t *size) {
  if (this->_iterations == 0 || this->_loop_idx < this->_data_num) {
    *data = _dataptr_list[_loop_idx % _dataptr_list.size()];
    *size = _data_size_list[_loop_idx % _data_size_list.size()];
    this->_loop_idx++;
    return true;
  }
  LOG_INFO << "All data feed.";
  return false;
}
