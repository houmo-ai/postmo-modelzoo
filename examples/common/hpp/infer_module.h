#include <iostream>
#include <sstream>
#include <string>

#include "tcim/tcim_runtime.h"

class InferModule {
 public:
  int Load(const std::string& model_path, tcim::Module::WeightManager& wm) {
    tcim::Module::Option option(wm);
    module_ = tcim::Module::LoadFromFile(model_path, option);
    if (module_.GetInitStatus() != tcim::OK) {
      return -1;
    }

    int input_num = module_.GetInputNum();
    std::cout << "Count of Input: " << input_num << std::endl;
    for (int idx = 0; idx < input_num; idx++) {
      auto input_name = module_.GetInputName(idx);
      auto input_info = module_.GetInputInfo(input_name);
      std::cout << "Input[" << input_name << "] " << input_info << std::endl;
      input_info_map_[input_name] = input_info;
    }

    int output_num = module_.GetOutputNum();
    std::cout << "Count of Output: " << output_num << std::endl;
    for (int idx = 0; idx < output_num; idx++) {
      auto output_name = module_.GetOutputName(idx);
      auto output_info = module_.GetOutputInfo(output_name);
      std::cout << "Output[" << output_name << "] " << output_info << std::endl;
      output_info_map_[output_name] = output_info;
    }
    return 0;
  }

  void SetInput(const std::string& name, const tcim::Tensor& input_tensor) {
    module_.SetInput(name, input_tensor);
  }

  void Run() {
    module_.Run();
  }

  void Sync() {
    module_.Sync();
  }

  void GetOutput(const std::string& name, tcim::Tensor& output_tensor) {
    module_.GetOutput(name, output_tensor);
  }

  tcim::Tensor GetOutput(const std::string& name) {
    return module_.GetOutput(name);
  }

  std::map<std::string, tcim::TensorInfo>& GetInputInfoMap() {
    return input_info_map_;
  }

  std::map<std::string, tcim::TensorInfo>& GetOutputInfoMap() {
    return output_info_map_;
  }

 protected:
  tcim::Module module_;
  std::map<std::string, tcim::TensorInfo> input_info_map_;
  std::map<std::string, tcim::TensorInfo> output_info_map_;
};