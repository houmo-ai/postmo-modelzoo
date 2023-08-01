// Copyright (c) 2022 The Houmo.ai Authors. All rights reserved.
#include <dmlc/json.h>
#include <tvm/runtime/executor_info.h>
#include <tvm/runtime/hdpl/hdpl_runtime.h>
#include <tvm/runtime/module.h>
#include <tvm/runtime/packed_func.h>
#include <tvm/runtime/registry.h>
#include <unistd.h>

#include <cassert>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "annotation.h"
#include "detect.h"
#include "hdpl/hdpl_runtime.h"
#include "hdpl_yolo_layer.h"

/**
 * @brief Load coco annotation file
 *
 * @param label_map  loaded map of image and label, (image_file_name, image
 * info)
 * @param file_names loaded image file vector
 * @param label_file_path label file path
 * @param max_count the count of images to be loaded
 * @return true load succeed
 * @return false load failed
 */
bool LoadAnnotationFile(std::unordered_map<std::string, ImageInfo> *label_map,
                        std::vector<std::string> *file_names,
                        const std::string &label_file_path, size_t max_count);

/**
 * @brief Load the preproccessed data file to input_map
 *
 * @param input_map output map of image name and preproccessed data
 * @param input_files data file vector
 * @param input_file_root_path preprocessed data root path
 * @return true loaded succeed
 * @return false loaded failed
 */
bool PreloadInputFile(
    std::unordered_map<std::string, tvm::runtime::NDArray> *input_map,
    const std::vector<std::string> &input_files,
    const std::string &input_file_root_path);

struct InputInfo {
  std::string input_name;
  std::vector<int64_t> input_shape;
  tvm::DataType input_dtype;
  size_t one_data_size;
};

/**
 * @brief Get the input name of model
 *
 * @param input_info output input information struct
 * @param module the houmo model pointer
 * @return true get input info succeed
 * @return false there is not only 1 input or no input
 */
bool HDPLRuntime_GetInputInfo(InputInfo *input_info, tvm::hdpl::Module *module);

/**
 * @brief Get the element count of shape
 *
 * @param shape tvm::runtime::ShapeTuple object
 * @return the element count
 */
size_t GetTensorElementCount(tvm::runtime::ShapeTuple shape) {
  size_t size = 1;
  for (size_t idx = 0; idx < shape.size(); ++idx) {
    size *= shape[idx];
  }
  return size;
}

/**
 * @brief post process of yolov3 model
 *
 * @param detections detection result vector
 * @param module the houmo model pointer
 * @param image_info the image info in annotation file
 * @param is_output0_big ture means the 0th output is 1 x 255 x 52 x 52, false
 * means the 0th is 1 x 255 x 13 x 13
 */
bool HDPLRuntime_CheckResult(std::vector<DetectInfo> *detections,
                             tvm::hdpl::Module *module,
                             const ImageInfo &image_info, bool is_output0_big);

int main(int argc, char *argv[]) {
  if (argc != 4) {
    std::cout << "Usage: " << argv[0] << " <annotation file path>"
              << " <input file root path>"
              << " <count>" << std::endl
              << "Example: " << argv[0]
              << " /nfsdata/datasets/COCO/annotations/instances_val2017.json"
              << " ./preprocessed"
              << " 10" << std::endl;
    return 1;
  }
  std::unordered_map<std::string, ImageInfo> image_info_map;
  std::vector<std::string> file_names;
  size_t count = std::atoi(argv[3]);
  std::string label_file_path(argv[1]);
  std::string input_file_root_path(argv[2]);
  if (!LoadAnnotationFile(&image_info_map, &file_names, label_file_path,
                          count)) {
    LOG(ERROR) << "Load label file " << label_file_path << " failed";
    return 1;
  }

  tvm::hdpl::Module module =
      tvm::hdpl::LoadModelPackage("../compile_model/yolov3");
  std::unordered_map<std::string, tvm::runtime::NDArray> input_map;
  if (!PreloadInputFile(&input_map, file_names, input_file_root_path)) {
    LOG(ERROR) << "Load input file failed.";
    return 1;
  }

  InputInfo input_info;
  if (!HDPLRuntime_GetInputInfo(&input_info, &module)) {
    LOG(ERROR) << "Get input info of the model failed";
    return -1;
  }
  std::vector<DetectInfo> detections;
  size_t output_ele_count_0 =
      GetTensorElementCount(module.GetFloatOutput(0).Shape());
  size_t output_ele_count_2 =
      GetTensorElementCount(module.GetFloatOutput(2).Shape());
  // Check the order of the following output, if the output0 is 1 x 255 x 52 x
  // 52, is_output0_big is true 1 x 255 x 52 x 52 output 1 x 255 x 26 x 26
  // output 1 x 255 x 13 x 13 output
  bool is_output0_big = false;
  if (output_ele_count_0 > output_ele_count_2) {
    is_output0_big = true;
  }
  for (std::string &input_file_name : file_names) {
    module.SetInput(input_info.input_name, input_map[input_file_name]);
    module.Run();
    hdplDeviceSynchronize();
    ImageInfo image_info = image_info_map[input_file_name];
    HDPLRuntime_CheckResult(&detections, &module, image_info, is_output0_big);
  }
  // Save detection file
  std::ofstream out_stream("detections.json");
  dmlc::JSONWriter writer(&out_stream);
  writer.Write(detections);
  out_stream.close();

  return 0;
}

bool LoadAnnotationFile(std::unordered_map<std::string, ImageInfo> *label_map,
                        std::vector<std::string> *file_names,
                        const std::string &label_file_path, size_t max_count) {
  std::ifstream json_file(label_file_path, std::ios::in);
  dmlc::JSONReader reader(&json_file);
  AnnotationInfo annoinfo;
  reader.Read(&annoinfo);
  size_t count = 0;
  for (ImageInfo &image_info : annoinfo.images) {
    if (count >= max_count) {
      break;
    }
    label_map->insert({image_info.file_name, image_info});
    file_names->push_back(image_info.file_name);
    count++;
  }
  return true;
}

bool PreloadInputFile(
    std::unordered_map<std::string, tvm::runtime::NDArray> *input_map,
    const std::vector<std::string> &input_files,
    const std::string &input_file_root_path) {
  for (std::string input_file_name : input_files) {
    std::string input_file_path = input_file_root_path + "/" + input_file_name;
    std::ifstream inFile(input_file_path, std::ios::in | std::ios::binary);
    tvm::runtime::NDArray image = tvm::runtime::NDArray::Empty(
        {1, 416, 416, 3}, tvm::DataType::Int(8), {kDLCPU, 0});
    int expected_file_size = 1 * 416 * 416 * 3;
    int file_size = expected_file_size;
    int offset = 0;
    char *data_ptr = reinterpret_cast<char *>(image->data);
    while (inFile.read(data_ptr + offset, file_size)) {
      int readedBytes = inFile.gcount();
      offset += readedBytes;
      file_size -= readedBytes;
      if (file_size < 0) {
        LOG(ERROR) << "load " << offset
                   << " datas from file: " << input_file_path << ", only "
                   << expected_file_size << " datas required.";
        return false;
      } else if (file_size == 0) {
        break;
      }
    }
    if (offset != expected_file_size) {
      LOG(ERROR) << "The size of file " << input_file_path
                 << " is not expected. " << offset << " vs "
                 << expected_file_size;
      return false;
    }
    input_map->insert({input_file_name, image});
  }
  LOG(INFO) << "Loaded " << input_map->size() << " files.";
  return true;
}

bool HDPLRuntime_GetInputInfo(InputInfo *input_info,
                              tvm::hdpl::Module *module) {
  int tvm_input_count = module->GetInputNum();
  bool found_input = false;
  for (int idx = 0; idx < tvm_input_count; idx++) {
    if (!module->IsParams(idx)) {
      if (found_input) {
        LOG(ERROR) << "Only one input supported in classification, but more "
                      "then one input found in the model. inputs: ("
                   << input_info->input_name << ", "
                   << module->GetInputNameByIndex(idx) << ")";
        return false;
      }
      input_info->input_name = module->GetInputNameByIndex(idx);
      found_input = true;
    }
  }
  if (!found_input) {
    LOG(ERROR) << "None input found in the model.";
    return false;
  }
  tvm::runtime::NDArray input_data =
      module->GetInputByName(input_info->input_name);
  auto input_shape = input_data.Shape();
  input_info->input_shape.clear();
  for (size_t idx = 0; idx < input_shape.size(); ++idx) {
    input_info->input_shape.push_back(input_shape[idx]);
  }
  input_info->input_dtype = input_data.DataType();
  size_t one_data_size = input_info->input_dtype.bytes();
  for (size_t dim_idx = 1; dim_idx < input_info->input_shape.size();
       dim_idx++) {
    one_data_size *= input_info->input_shape[dim_idx];
  }
  input_info->one_data_size = one_data_size;
  return true;
}

void SaveNdarray(std::string file_name, tvm::runtime::NDArray ndarray) {
  auto cpu_tensor = const_cast<DLTensor *>(ndarray.operator->());
  char *big_out = static_cast<char *>(cpu_tensor->data);
  std::ofstream outfile(file_name, std::ifstream::binary);
  outfile.write(big_out, GetTensorElementCount(ndarray.Shape()));
  outfile.close();
}

bool HDPLRuntime_CheckResult(std::vector<DetectInfo> *detections,
                             tvm::hdpl::Module *module,
                             const ImageInfo &image_info, bool is_output0_big) {
  // Get pointer to output tensor float values
  // 1 x 255 x 52 x 52 output
  // SaveNdarray("1_255_52_52", module->GetOutput(is_output0_big ? 0 : 2));
  auto output_0 = module->GetFloatOutput(is_output0_big ? 0 : 2);
  auto cpu_tensor_0 = const_cast<DLTensor *>(output_0.operator->());
  float *big_out = static_cast<float *>(cpu_tensor_0->data);

  // 1 x 255 x 26 x 26 output
  // SaveNdarray("1_255_26_26", module->GetOutput(1));

  auto output_1 = module->GetFloatOutput(1);
  auto cpu_tensor_1 = const_cast<DLTensor *>(output_1.operator->());
  float *mid_out = static_cast<float *>(cpu_tensor_1->data);

  // 1 x 255 x 13 x 13 output
  // SaveNdarray("1_255_13_13", module->GetOutput(is_output0_big ? 2 : 0));
  auto output_2 = module->GetFloatOutput(is_output0_big ? 2 : 0);
  auto cpu_tensor_2 = const_cast<DLTensor *>(output_2.operator->());
  float *small_out = static_cast<float *>(cpu_tensor_2->data);

  if (!yolo_detect(detections, big_out, mid_out, small_out, image_info)) {
    std::cerr << "Yolo failed" << std::endl;
    return false;
  }
  return true;
}
