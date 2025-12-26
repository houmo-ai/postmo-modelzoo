#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <string>

#ifdef __cplusplus
extern "C" {
#endif
#include "hm_sys.h"
#ifdef __cplusplus
}
#endif

int main(int argc, char** argv) {
  int device_id = -1;
  std::string output_file;

  int opt;
  while ((opt = getopt(argc, argv, "hd:o:")) != -1) {
    switch (opt) {
      case 'h':
        std::cout << "用法: " << argv[0] << " [选项]\n"
                  << "  -h    显示帮助信息\n"
                  << "  -d    (可选)监测的设备ID, 默认监测第一个设备\n";
        return 0;

      case 'd':
        device_id = std::stoi(optarg);
        break;

      case '?':  // 未知选项或缺少参数
        // optopt 存储未知的选项
        std::cerr << "错误：未知选项 '" << char(optopt) << "' 或缺少参数\n";
        return 1;

      default:
        return 1;
    }
  }

  hm_device_info dev_info = {0};
  int ret = hm_sys_get_device_info(&dev_info);

  if (ret <= 0 || dev_info.num_devices <= 0) {
    std::cerr << "Not found online devices." << std::endl;
    return -1;
  }
  std::cout << "Online device num: " << dev_info.num_devices
            << ", online deivce id: ";
  for (int i = 0; i < dev_info.num_devices; i++) {
    std::cout << dev_info.device_ids[i] << " ";
  }
  std::cout << std::endl;

  if (device_id >= 0) {
    bool inCArr = std::find(std::begin(dev_info.device_ids),
                            std::end(dev_info.device_ids),
                            device_id) != std::end(dev_info.device_ids);
    if (!inCArr) {
      std::cerr << "Invalid device id " << device_id << std::endl;
      return -1;
    }
  }

  if (device_id < 0) {
    device_id = dev_info.device_ids[0];
  }
  hm_mem_info mem_info = {0};
  auto now = std::chrono::system_clock::now();
  std::time_t current_time = std::chrono::system_clock::to_time_t(now);
  std::tm* local_time = std::localtime(&current_time);

  ret = hm_sys_get_mem_info(device_id, &mem_info);
  std::cout << "device_id: " << device_id
            << ", time: " << std::put_time(local_time, "%Y-%m-%d %H:%M:%S")
            << ", mem_total: " << mem_info.mem_total
            << ", mem_used: " << mem_info.mem_used
            << ", mem_avail: " << mem_info.mem_avail << std::endl;

  return 0;
}