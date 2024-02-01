#ifndef CV_YOLOV3_INFERENCE_MODEL_ANNOTATION_H_
#define CV_YOLOV3_INFERENCE_MODEL_ANNOTATION_H_

#include <algorithm>
#include <string>
#include <vector>

#include "dmlc/json.h"

struct ImageInfo {
  std::string file_name;
  int height;
  int width;
  int id;

  void Load(dmlc::JSONReader *reader) {
    reader->BeginObject();
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "file_name") {
        reader->Read(&file_name);
      } else if (key == "height") {
        reader->Read(&height);
      } else if (key == "width") {
        reader->Read(&width);
      } else if (key == "id") {
        reader->Read(&id);
      } else if (key == "license") {
        int32_t skipped_int;
        reader->Read(&skipped_int);
      } else if (key == "coco_url" || key == "date_captured" ||
                 key == "flickr_url") {
        std::string skipped_str;
        reader->Read(&skipped_str);
      } else {
        LOG(ERROR) << "Usupported key: " << key;
      }
    }
  }
};

struct IgnoreInfo {
  std::vector<std::string> int_keys;
  std::vector<std::string> str_keys;
  std::vector<std::string> float_keys;
  std::vector<std::string> vec_keys;
  std::vector<std::string> vec2d_keys;

  IgnoreInfo(const std::vector<std::string> &i_keys,
             const std::vector<std::string> &s_keys,
             const std::vector<std::string> &f_keys,
             const std::vector<std::string> &v_keys) {
    this->int_keys = i_keys;
    this->str_keys = s_keys;
    this->float_keys = f_keys;
    this->vec_keys = v_keys;
  }

  IgnoreInfo(const std::vector<std::string> &i_keys,
             const std::vector<std::string> &s_keys,
             const std::vector<std::string> &f_keys,
             const std::vector<std::string> &v_keys,
             const std::vector<std::string> &v2d_keys) {
    this->int_keys = i_keys;
    this->str_keys = s_keys;
    this->float_keys = f_keys;
    this->vec_keys = v_keys;
    this->vec2d_keys = v2d_keys;
  }

  void Load(dmlc::JSONReader *reader) {
    reader->BeginObject();
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (std::find(int_keys.begin(), int_keys.end(), key) != int_keys.end()) {
        int32_t skipped_int;
        reader->Read(&skipped_int);
      } else if (std::find(str_keys.begin(), str_keys.end(), key) !=
                 str_keys.end()) {
        std::string skipped_str;
        reader->Read(&skipped_str);
      } else if (std::find(float_keys.begin(), float_keys.end(), key) !=
                 float_keys.end()) {
        float skipped_float;
        reader->Read(&skipped_float);
      } else if (std::find(vec_keys.begin(), vec_keys.end(), key) !=
                 vec_keys.end()) {
        reader->BeginArray();
        float skipped_float;
        while (reader->NextArrayItem()) {
          reader->Read(&skipped_float);
        }
      } else if (std::find(vec2d_keys.begin(), vec2d_keys.end(), key) !=
                 vec2d_keys.end()) {
        reader->BeginArray();
        float skipped_float;
        while (reader->NextArrayItem()) {
          reader->BeginArray();
          while (reader->NextArrayItem()) {
            reader->Read(&skipped_float);
          }
        }
      } else {
        LOG(ERROR) << "Unsupported key: " << key;
      }
    }
  }
};

struct AnnotationInfo {
  AnnotationInfo() {}
  std::vector<ImageInfo> images;

  void Load(dmlc::JSONReader *reader) {
    reader->BeginObject();
    std::string key;
    while (reader->NextObjectItem(&key)) {
      if (key == "images") {
        reader->Read(&images);
        // break to ignore rest fields
        break;
      } else if (key == "info") {
        IgnoreInfo info(
            {"year"},
            {"description", "url", "version", "contributor", "date_created"},
            {}, {});
        reader->Read(&info);
      } else if (key == "licenses") {
        reader->BeginArray();
        IgnoreInfo info({"id"}, {"url", "name"}, {}, {});
        while (reader->NextArrayItem()) {
          reader->Read(&info);
        }
      } else if (key == "annotations") {
        // some segmentation settings get error
        reader->BeginArray();
        IgnoreInfo info({"iscrowd", "image_id", "category_id", "id"}, {},
                        {"area"}, {"bbox"}, {"segmentation"});
        while (reader->NextArrayItem()) {
          reader->Read(&info);
        }
      } else if (key == "categories") {
        reader->BeginArray();
        IgnoreInfo info({"id"}, {"supercategory", "name"}, {}, {});
        while (reader->NextArrayItem()) {
          reader->Read(&info);
        }
      } else {
        LOG(ERROR) << "Unsupported key: " << key;
      }
    }
  }
};

#endif  // CV_YOLOV3_INFERENCE_MODEL_ANNOTATION_H_
