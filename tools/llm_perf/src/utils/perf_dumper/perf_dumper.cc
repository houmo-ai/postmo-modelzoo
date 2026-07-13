/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: perf_dumper.cc
 * Description:
 *   perf_dumper Implementation - dump perf metrics to yaml file.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 */
#include "utils/perf_dumper/perf_dumper.h"

#include <cmath>
#include <cstdlib>

#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/spdlog.h"

namespace {
std::string format_device_memory(uint32_t value_mb) {
  return std::to_string(value_mb) + " MB";
}

std::string format_device_memory(int value_mb) {
  return std::to_string(value_mb) + " MB";
}

std::string format_device_memory(double value_mb) {
  return format_double(value_mb) + " MB";
}

int format_device_memory_avg_as_int(double value_mb) {
  return static_cast<int>(std::lround(value_mb));
}
}  // namespace

PerfDumper::PerfDumper() : dump_file("") {}

PerfDumper::~PerfDumper() = default;

void PerfDumper::setYamlFile(const std::string &yaml_file, bool run_yaml_perf) {
  if (dump_file.empty()) {
    dump_file = yaml_file;
    root.reset();
    init_yaml = true;
  } else {
    if (!run_yaml_perf) {
      dump_file = yaml_file;
      root.reset();
      init_yaml = true;
    }
  }
}

void PerfDumper::dumpPerf(
    const PerfSettings &perf_settings,
    const InferenceMetricsWithLoadTime &results,
    const HostMemoryInfo &host_mem_info,
    const HostMemoryInfo &max_host_mem_info,
    const std::unordered_map<int, DeviceStats> &post_init_dev_stats,
    const std::unordered_map<int, DeviceStats> &end_device_stats) {
  if (dump_file.empty()) return;
  if (init_yaml) {
    root.reset();
    root["PerfMetrics"] = YAML::Node(YAML::NodeType::Sequence);
    init_yaml = false;
  } else if (!root["PerfMetrics"] || !root["PerfMetrics"].IsSequence()) {
    root["PerfMetrics"] = YAML::Node(YAML::NodeType::Sequence);
  }

  YAML::Node perf_metrics_node = root["PerfMetrics"];

  YAML::Node model_metrics;
  YAML::Node model_perf_settings = model_metrics["PerfSettings"];
  model_perf_settings["ModelName"] = perf_settings.model_name;
  model_perf_settings["prefill"] = perf_settings.prefill_path;
  model_perf_settings["decode"] = perf_settings.decode_path;
  model_perf_settings["visual"] = perf_settings.visual_path;
  model_perf_settings["embedding"] = perf_settings.embedding_path;
  model_perf_settings["input"] = perf_settings.input_tokens_len;
  model_perf_settings["output"] = perf_settings.stop_tokens_len;
  YAML::Node devices_node;
  for (int d : perf_settings.devices) {
    devices_node.push_back(d);
  }
  model_perf_settings["devices"] = devices_node;
  model_perf_settings["batch"] = perf_settings.batch_size;
  model_perf_settings["loop"] = perf_settings.loop_count;
  model_perf_settings["LazyMode"] = perf_settings.LazyMode;
  model_perf_settings["warm_up"] = perf_settings.warm_up;
  model_perf_settings["warm_up_input"] = perf_settings.warm_up_input;
  model_perf_settings["warm_up_output"] = perf_settings.warm_up_output;
  model_perf_settings["skip_perf"] = perf_settings.skip_perf;
  model_perf_settings["monitor_interval"] = perf_settings.interval_ms;
  model_perf_settings["perf_case_index"] = perf_settings.perf_case_index;
  model_perf_settings["perf_case_total"] = perf_settings.perf_case_total;

  YAML::Node model_perf_results = model_metrics["PerfResults"];
  model_perf_results["input_token"] = results.metrics.input_seq_length;
  model_perf_results["output_token"] = results.metrics.output_seq_length;
  model_perf_results["prefill_load_time"] =
      format_double(results.prefill_load_time);
  model_perf_results["decode_load_time"] =
      format_double(results.decode_load_time);
  model_perf_results["vision_load_time"] =
      format_double(results.vision_load_time);
  model_perf_results["prefill_time"] =
      format_double(results.metrics.prefill_perf_infos.total_time);
  model_perf_results["decode_time"] =
      format_double(results.metrics.decode_perf_infos.total_time);
  model_perf_results["vision_time"] =
      format_double(results.metrics.vision_perf_infos.vision_total_time);
  model_perf_results["prefill_speed"] =
      format_double(results.metrics.prefill_perf_infos.total_speed);
  model_perf_results["decode_speed"] =
      format_double(results.metrics.decode_perf_infos.total_speed);
  model_perf_results["vision_speed"] =
      format_double(results.metrics.vision_perf_infos.vision_total_speed);
  model_perf_results["prefill_infer_time_avg"] =
      format_double(results.metrics.prefill_perf_infos.infer_time /
                    perf_settings.input_tokens_len);
  model_perf_results["decode_infer_time_avg"] =
      format_double(results.metrics.decode_perf_infos.infer_time_per_token);
  model_perf_results["vision_infer_time_avg"] =
      format_double(results.metrics.vision_perf_infos.infer_time);
  model_perf_results["prefill_infer_speed_avg"] =
      format_double(results.metrics.prefill_perf_infos.infer_speed);
  model_perf_results["decode_infer_speed_avg"] =
      format_double(results.metrics.decode_perf_infos.infer_speed);
  model_perf_results["vision_infer_speed_avg"] =
      format_double(results.metrics.vision_perf_infos.vision_infer_speed);
  model_perf_results["prefill_embedding_time"] =
      format_double(results.metrics.prefill_perf_infos.embedding_time);
  model_perf_results["decode_embedding_time"] =
      format_double(results.metrics.decode_perf_infos.embedding_time);
  model_perf_results["kvcache_mem"] = format_double(results.kvcache_mem);
  model_perf_results["TTFT"] = format_double(results.metrics.ttft);
  model_perf_results["TPOT"] = format_double(results.metrics.tpot);
  model_perf_results["e2e_latency"] = format_double(results.metrics.e2e_time);
  model_perf_results["e2e_tps"] = format_double(results.metrics.e2e_tps);

  // Host Monitor
#if defined(__linux__)
  YAML::Node host_metrics = model_metrics["HostMonitor"];
  host_metrics["physical_memory"] =
      formatHostMemorySize(host_mem_info.physical_memory);
  host_metrics["virtual_memory"] =
      formatHostMemorySize(host_mem_info.virtual_memory);
  host_metrics["max_physical_memory"] =
      formatHostMemorySize(max_host_mem_info.physical_memory);
  host_metrics["max_virtual_memory"] =
      formatHostMemorySize(max_host_mem_info.virtual_memory);
#endif

  YAML::Node device_metrics = model_metrics["DeviceMonitor"];
  for (auto &[dev_id, device_stats] : end_device_stats) {
    YAML::Node device_metrics_node = device_metrics[std::to_string(dev_id)];
    device_metrics_node["ipu_freq_max"] =
        format_double(device_stats.ipu_freq_max) + " MHz";
    device_metrics_node["ipu_freq_min"] =
        format_double(device_stats.ipu_freq_min) + " MHz";
    device_metrics_node["ipu_freq_avg"] =
        format_double(device_stats.ipu_freq_avg) + " MHz";
    device_metrics_node["temperature_max"] =
        format_double(device_stats.temperature_max) + " °C";
    device_metrics_node["temperature_min"] =
        format_double(device_stats.temperature_min) + " °C";
    device_metrics_node["temperature_avg"] =
        format_double(device_stats.temperature_avg) + " °C";
    device_metrics_node["power_max"] =
        format_double(device_stats.power_max) + " W";
    device_metrics_node["power_min"] =
        format_double(device_stats.power_min) + " W";
    device_metrics_node["power_avg"] =
        format_double(device_stats.power_avg) + " W";
    device_metrics_node["mem_total"] =
        format_device_memory(device_stats.mem_info.mem_total);
    device_metrics_node["mem_used"] =
        format_device_memory(device_stats.mem_info.mem_used);
    device_metrics_node["mem_used_max"] =
        format_device_memory(device_stats.mem_used_max);
    device_metrics_node["mem_used_min"] =
        format_device_memory(device_stats.mem_used_min);
    device_metrics_node["mem_used_avg"] = format_device_memory(
        format_device_memory_avg_as_int(device_stats.mem_used_avg));
  }

  YAML::Node model_load_metrics = model_metrics["ModelLoadMemory"];
  for (const auto &[dev_id, init_end_stats] : post_init_dev_stats) {
    YAML::Node model_load_metrics_node =
        model_load_metrics[std::to_string(dev_id)];
    model_load_metrics_node["mem_total"] =
        format_device_memory(init_end_stats.mem_info.mem_total);
    model_load_metrics_node["mem_used"] =
        format_device_memory(init_end_stats.mem_info.mem_used);
  }
  perf_metrics_node.push_back(model_metrics);
}

#ifdef ENABLE_ASR
void PerfDumper::dumpAsrPerf(
    const AsrPerfSettings &perf_settings,
    const AsrTranscribeResult &results,
    int n_chunks,
    const HostMemoryInfo &host_mem_info,
    const HostMemoryInfo &max_host_mem_info,
    const std::unordered_map<int, DeviceStats> &post_init_dev_stats,
    const std::unordered_map<int, DeviceStats> &end_device_stats) {
  if (dump_file.empty()) return;
  if (init_yaml) {
    root.reset();
    root["PerfMetrics"] = YAML::Node(YAML::NodeType::Sequence);
    init_yaml = false;
  } else if (!root["PerfMetrics"] || !root["PerfMetrics"].IsSequence()) {
    root["PerfMetrics"] = YAML::Node(YAML::NodeType::Sequence);
  }

  YAML::Node perf_metrics_node = root["PerfMetrics"];

  YAML::Node model_metrics;
  YAML::Node model_perf_settings = model_metrics["PerfSettings"];
  model_perf_settings["ModelName"] = perf_settings.model_name;
  model_perf_settings["encode"] = perf_settings.encode_path;
  model_perf_settings["prefill"] = perf_settings.prefill_path;
  model_perf_settings["decode"] = perf_settings.decode_path;
  model_perf_settings["chunk"] = perf_settings.chunk;
  model_perf_settings["audio_len"] = perf_settings.audio_len_seconds;
  model_perf_settings["token_per_second"] = perf_settings.token_per_second;
  YAML::Node devices_node;
  for (int d : perf_settings.devices) {
    devices_node.push_back(d);
  }
  model_perf_settings["devices"] = devices_node;
  model_perf_settings["loop"] = perf_settings.loop_count;
  model_perf_settings["warm_up"] = perf_settings.warm_up;
  model_perf_settings["monitor_interval"] = perf_settings.interval_ms;
  model_perf_settings["perf_case_index"] = perf_settings.perf_case_index;
  model_perf_settings["perf_case_total"] = perf_settings.perf_case_total;

  YAML::Node model_perf_results = model_metrics["PerfResults"];
  model_perf_results["audio_duration_s"] = results.audio_duration_s;
  model_perf_results["chunks"] = n_chunks;
  model_perf_results["output_tokens"] = results.output_tokens;
  model_perf_results["encode_time"] = format_double(results.encode_time_ms);
  model_perf_results["prefill_time"] = format_double(results.prefill_time_ms);
  model_perf_results["decode_time"] = format_double(results.decode_time_ms);
  model_perf_results["total_time"] = format_double(results.total_time_ms);
  model_perf_results["TTFT"] = format_double(results.ttft_ms);
  model_perf_results["overall_rtf"] = format_double(results.overall_rtf, 4);
  model_perf_results["inference_rtf"] = format_double(results.inference_rtf, 4);
  model_perf_results["decode_tps"] = format_double(results.decode_tps);
  model_perf_results["overall_tps"] = format_double(results.overall_tps);

#if defined(__linux__)
  YAML::Node host_metrics = model_metrics["HostMonitor"];
  host_metrics["physical_memory"] =
      formatHostMemorySize(host_mem_info.physical_memory);
  host_metrics["virtual_memory"] =
      formatHostMemorySize(host_mem_info.virtual_memory);
  host_metrics["max_physical_memory"] =
      formatHostMemorySize(max_host_mem_info.physical_memory);
  host_metrics["max_virtual_memory"] =
      formatHostMemorySize(max_host_mem_info.virtual_memory);
#endif

  YAML::Node device_metrics = model_metrics["DeviceMonitor"];
  for (auto &[dev_id, device_stats] : end_device_stats) {
    YAML::Node device_metrics_node = device_metrics[std::to_string(dev_id)];
    device_metrics_node["ipu_freq_max"] =
        format_double(device_stats.ipu_freq_max) + " MHz";
    device_metrics_node["ipu_freq_min"] =
        format_double(device_stats.ipu_freq_min) + " MHz";
    device_metrics_node["ipu_freq_avg"] =
        format_double(device_stats.ipu_freq_avg) + " MHz";
    device_metrics_node["temperature_max"] =
        format_double(device_stats.temperature_max) + " °C";
    device_metrics_node["temperature_min"] =
        format_double(device_stats.temperature_min) + " °C";
    device_metrics_node["temperature_avg"] =
        format_double(device_stats.temperature_avg) + " °C";
    device_metrics_node["power_max"] =
        format_double(device_stats.power_max) + " W";
    device_metrics_node["power_min"] =
        format_double(device_stats.power_min) + " W";
    device_metrics_node["power_avg"] =
        format_double(device_stats.power_avg) + " W";
    device_metrics_node["mem_total"] =
        format_device_memory(device_stats.mem_info.mem_total);
    device_metrics_node["mem_used"] =
        format_device_memory(device_stats.mem_info.mem_used);
    device_metrics_node["mem_used_max"] =
        format_device_memory(device_stats.mem_used_max);
    device_metrics_node["mem_used_min"] =
        format_device_memory(device_stats.mem_used_min);
    device_metrics_node["mem_used_avg"] = format_device_memory(
        format_device_memory_avg_as_int(device_stats.mem_used_avg));
  }

  YAML::Node model_load_metrics = model_metrics["ModelLoadMemory"];
  for (const auto &[dev_id, init_end_stats] : post_init_dev_stats) {
    YAML::Node model_load_metrics_node =
        model_load_metrics[std::to_string(dev_id)];
    model_load_metrics_node["mem_total"] =
        format_device_memory(init_end_stats.mem_info.mem_total);
    model_load_metrics_node["mem_used"] =
        format_device_memory(init_end_stats.mem_info.mem_used);
  }
  perf_metrics_node.push_back(model_metrics);
}
#endif

void PerfDumper::showPerfBrief(
    const PerfSettings &perf_settings,
    const InferenceMetricsWithLoadTime &results,
    const HostMemoryInfo &host_mem_info,
    const HostMemoryInfo &max_host_mem_info,
    const std::unordered_map<int, DeviceStats> &post_init_dev_stats,
    const std::unordered_map<int, DeviceStats> &end_device_stats) {
  auto metrics = results.metrics;
  std::cout << COLOR_MAGENT << std::fixed << std::setprecision(2);
  std::cout << "\n" << std::string(82, '=') << std::endl;
  std::cout << "                    Model Inference Performance Summary Report"
            << std::endl;
  std::cout << std::string(82, '-') << std::endl;
  // Basic configuration
  std::cout << "                            Configuration Details" << std::endl;
  std::cout << std::string(82, '-') << std::endl;
  if (perf_settings.perf_case_total > 1) {
    std::cout << "  Perf Case: " << std::setw(6)
              << perf_settings.perf_case_index << "/"
              << perf_settings.perf_case_total << std::endl;
  }
  std::cout << "  Batch Size: " << std::setw(6) << metrics.batch_size
            << std::endl;
  std::cout << "  Input Length per Sample: " << std::setw(6)
            << metrics.input_seq_length << " tokens" << std::endl;
  std::cout << "  Output Length per Sample: " << std::setw(6)
            << (metrics.output_seq_length) << " tokens" << std::endl;
  if (metrics.num_images > 0) {
    std::cout << "  Number of Images: " << std::setw(6) << metrics.num_images
              << " images" << std::endl;
  }
  std::cout << std::string(82, '-') << std::endl;
  // Brief Inference Performance
  std::cout << "                            Inference Performance "
            << std::endl;
  std::cout << std::string(82, '-') << std::endl;
  std::cout << "  Prefill API Inference total Time: "
            << results.metrics.prefill_perf_infos.infer_time << std::setw(5)
            << " ms | Speed: " << results.metrics.prefill_perf_infos.infer_speed
            << " tokens/s" << std::endl;
  std::cout << "  Decode  API Inference total Time: "
            << results.metrics.decode_perf_infos.infer_time << std::setw(5)
            << " ms | Speed: " << results.metrics.decode_perf_infos.infer_speed
            << " tokens/s" << std::endl;
  std::cout << "  Vision  API Inference total Time: "
            << results.metrics.vision_perf_infos.infer_time << std::setw(5)
            << " ms | Speed: "
            << results.metrics.vision_perf_infos.vision_infer_speed
            << " images/s" << std::endl;

  // Summary metrics
  std::cout << std::string(82, '-') << std::endl;
  std::cout << "                            Overall Performance " << std::endl;
  std::cout << std::string(82, '-') << std::endl;
  if (results.prefill_load_time > 0) {
    std::cout << "  Prefill Model Load Time: " << std::setw(7)
              << results.prefill_load_time << "ms" << std::endl;
  }
  if (results.decode_load_time > 0) {
    std::cout << "  Decode Model Load Time: " << std::setw(7)
              << results.decode_load_time << "ms" << std::endl;
  }
  if (results.prefill_load_time > 0) {
    std::cout << "  Vision Model Load Time: " << std::setw(7)
              << results.vision_load_time << "ms" << std::endl;
  }
  if (!post_init_dev_stats.empty()) {
    std::cout << "  Model Load Device Memory:" << std::endl;
    for (const auto &[dev_id, device_stats] : post_init_dev_stats) {
      std::cout << "    Device " << dev_id
                << " | Total: " << device_stats.mem_info.mem_total << " MB"
                << " | Used: " << device_stats.mem_info.mem_used << " MB"
                << std::endl;
    }
  }
  std::cout << "  TTFT (Time To First Token): " << std::setw(7) << metrics.ttft
            << " ms" << std::endl;
  std::cout << "  TPOT (Time Per Output Token): " << std::setw(5)
            << metrics.tpot << " ms/token" << std::endl;
  std::cout << "  E2E Latency (End-to-End): " << std::setw(9)
            << metrics.e2e_time << " seconds" << std::endl;
  std::cout << "  E2E TPS (Throughput): " << std::setw(13) << metrics.e2e_tps
            << " tokens/s" << std::endl;

  std::cout << std::string(82, '-') << std::endl;
  // host info
#if defined(__linux__)
  std::cout << "                            Memory Usage (Max Values)"
            << std::endl;
  std::cout << std::string(82, '-') << std::endl;
  std::cout << "  Physical Memory: "
            << formatMemorySize(host_mem_info.physical_memory) << std::endl;
  std::cout << "  Virtual Memory: "
            << formatMemorySize(host_mem_info.virtual_memory) << std::endl;
  std::cout << "  Max Physical Memory: "
            << formatMemorySize(max_host_mem_info.physical_memory) << std::endl;
  std::cout << "  Max Virtual Memory: "
            << formatMemorySize(max_host_mem_info.virtual_memory) << std::endl;
#endif
  // device info
  std::cout << std::string(82, '-') << std::endl;
  std::cout << "                            Device Stats (Max Values)  "
            << std::endl;
  std::cout << std::string(82, '-') << std::endl;
  for (const auto &[dev_id, device_stats] : end_device_stats) {
    std::cout << "Device " << dev_id << ": " << std::endl;
    auto fmt = [](auto v, int prec, const char *unit) -> std::string {
      std::ostringstream o;
      o << std::fixed << std::setprecision(prec) << v << unit;
      return o.str();
    };
    std::cout << std::left << std::setw(15) << "Temperature"
              << "|  " << std::left << std::setw(18)
              << fmt(device_stats.temperature_min, 2, "°C(Min)") << " |  "
              << std::left << std::setw(18)
              << fmt(device_stats.temperature_max, 2, "°C(Max)") << " |  "
              << std::left << std::setw(18)
              << fmt(device_stats.temperature_avg, 2, "°C(Avg)") << " |"
              << std::endl;
    std::cout << std::left << std::setw(15) << "Power"
              << "|  " << std::left << std::setw(18)
              << fmt(device_stats.power_min, 2, " W(Min)") << "|  " << std::left
              << std::setw(18) << fmt(device_stats.power_max, 2, " W(Max)")
              << "|  " << std::left << std::setw(18)
              << fmt(device_stats.power_avg, 2, " W(Avg)") << "|" << std::endl;

    std::cout << std::left << std::setw(15) << "IPU Freq"
              << "|  " << std::left << std::setw(18)
              << fmt(device_stats.ipu_freq_min, 2, " Mhz(Min)") << "|  "
              << std::left << std::setw(18)
              << fmt(device_stats.ipu_freq_max, 2, " Mhz(Max)") << "|  "
              << std::left << std::setw(18)
              << fmt(device_stats.ipu_freq_avg, 2, " Mhz(Avg)") << "|"
              << std::endl;

    std::cout << std::left << std::setw(15) << "Mem Info"
              << "|  " << std::left << std::setw(18)
              << (std::to_string(device_stats.mem_info.mem_total) +
                  " MB(Total)")
              << "|  " << std::left << std::setw(18)
              << (std::to_string(device_stats.mem_info.mem_used) + " MB(Used)")
              << "|  " << std::left << std::setw(18)
              << (std::to_string(device_stats.mem_info.mem_avail) +
                  " MB(Avail)")
              << "|" << std::endl;
    std::cout << std::left << std::setw(15) << "Mem Used"
              << "|  " << std::left << std::setw(18)
              << fmt(device_stats.mem_used_min, 2, " MB(Min)") << "|  "
              << std::left << std::setw(18)
              << fmt(device_stats.mem_used_max, 2, " MB(Max)") << "|  "
              << std::left << std::setw(18)
              << fmt(format_device_memory_avg_as_int(device_stats.mem_used_avg),
                     0, " MB(Avg)")
              << "|" << std::endl;
  }
  std::cout << std::string(82, '-') << std::endl;
  // Show Detail Model Metrics

  // Vision stage performance (if any)
  if (metrics.num_images > 0 &&
      (metrics.vision_perf_infos.vision_total_time > 0 ||
       metrics.vision_perf_infos.vision_preprocess_time > 0)) {
    std::cout << "                            Vision Stage Performance "
              << std::endl;
    std::cout << std::string(82, '-') << std::endl;
    std::cout << "  Total Time: " << std::setw(7)
              << metrics.vision_perf_infos.vision_total_time
              << "ms | Speed: " << std::setw(7)
              << metrics.vision_perf_infos.vision_total_speed << " images/s"
              << std::endl;
    if (metrics.vision_perf_infos.vision_preprocess_time > 0) {
      std::cout << "  Preprocessing Time: " << std::setw(5)
                << metrics.vision_perf_infos.vision_preprocess_time
                << "ms | Speed: " << std::setw(7)
                << metrics.vision_perf_infos.vision_preprocess_speed
                << " images/s" << std::endl;
    } else {
      std::cout << "  Preprocessing Time: Skipped (No operation)" << std::endl;
    }

    std::cout << "  API SetInput  total Time: " << std::setw(6)
              << metrics.vision_perf_infos.setinput_time << "ms" << std::endl;
    std::cout << "  API Inference total Time: " << std::setw(5)
              << metrics.vision_perf_infos.infer_time
              << "ms | Speed: " << std::setw(7)
              << metrics.vision_perf_infos.vision_infer_speed << " images/s"
              << std::endl;
    std::cout << "  API GetOutput total Time: " << std::setw(5)
              << metrics.vision_perf_infos.getoutput_time << "ms" << std::endl;
    std::cout << std::string(82, '-') << std::endl;
  }

  // Prefill stage performance
  std::cout << "                            Prefill Stage Performance "
            << std::endl;
  std::cout << std::string(82, '-') << std::endl;

  std::cout << "  Total Time: " << std::setw(7)
            << metrics.prefill_perf_infos.total_time
            << "ms | Speed: " << std::setw(7)
            << metrics.prefill_perf_infos.total_speed << " tokens/s"
            << std::endl;
  if (metrics.prefill_perf_infos.tokenizer_time > 0) {
    std::cout << "  Tokenization total Time: " << std::setw(7)
              << metrics.prefill_perf_infos.tokenizer_time << "ms" << std::endl;
  } else {
    std::cout << "  Tokenization total Time: Skipped (No operation)"
              << std::endl;
  }
  std::cout << "  Embedding total Time: " << std::setw(7)
            << metrics.prefill_perf_infos.embedding_time << "ms" << std::endl;
  std::cout << "  API SetInput  total Time: " << std::setw(6)
            << metrics.prefill_perf_infos.setinput_time << "ms" << std::endl;
  std::cout << "  API Inference total Time: " << std::setw(5)
            << metrics.prefill_perf_infos.infer_time
            << "ms | Speed: " << std::setw(7)
            << metrics.prefill_perf_infos.infer_speed << " tokens/s"
            << std::endl;
  std::cout << "  API GetOutput total Time: " << std::setw(5)
            << metrics.prefill_perf_infos.getoutput_time << "ms" << std::endl;
  std::cout << std::string(82, '-') << std::endl;

  // Decode stage performance
  std::cout << "                            Decode Stage Performance "
            << std::endl;
  std::cout << std::string(82, '-') << std::endl;
  std::cout << "  Total Time: " << std::setw(7)
            << metrics.decode_perf_infos.total_time
            << "ms | Speed: " << std::setw(7)
            << metrics.decode_perf_infos.total_speed << " tokens/s"
            << std::endl;
  if (metrics.decode_perf_infos.tokenizer_time > 0) {
    std::cout << "  Tokenization total Time: " << std::setw(7)
              << metrics.decode_perf_infos.tokenizer_time << "ms" << std::endl;
  } else {
    std::cout << "  Tokenization total Time: Skipped (No operation)"
              << std::endl;
  }
  std::cout << "  Embedding total Time: " << std::setw(7)
            << metrics.decode_perf_infos.embedding_time << "ms" << std::endl;
  std::cout << "  API SetInput  avg Time: " << std::setw(6)
            << metrics.decode_perf_infos.setinput_time_per_token << "ms/token"
            << std::endl;
  std::cout << "  API Inference avg Time: " << std::setw(5)
            << metrics.decode_perf_infos.infer_time_per_token
            << "ms/token | Speed: " << std::setw(7)
            << metrics.decode_perf_infos.infer_speed << " tokens/s"
            << std::endl;
  std::cout << "  API GetOutput avg Time: " << std::setw(5)
            << metrics.decode_perf_infos.getoutput_time_per_token << "ms/token"
            << std::endl;

  std::cout << std::string(82, '=') << COLOR_RESET << std::endl;
}

void PerfDumper::writePerfBrief(
    const PerfSettings &perf_settings,
    const InferenceMetricsWithLoadTime &results,
    const HostMemoryInfo &host_mem_info,
    const HostMemoryInfo &max_host_mem_info,
    const std::unordered_map<int, DeviceStats> &post_init_dev_stats,
    const std::unordered_map<int, DeviceStats> &end_device_stats,
    std::string perf_intruduction) {
  auto metrics = results.metrics;

  static auto logger =
      spdlog::rotating_logger_mt("perf_logger", log_file, 1024 * 1024 * 5, 3);
  logger->set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%l] %v");

  // Basic configuration

  logger->info(perf_intruduction);
  logger->info(
      "==================== Model Inference Performance Summary Report "
      "====================");
  logger->info(
      "-------------------- Configuration Details ---------------------");
  if (perf_settings.perf_case_total > 1) {
    logger->info("  Perf Case: {:>6}/{}", perf_settings.perf_case_index,
                 perf_settings.perf_case_total);
  }
  logger->info("  Batch Size: {:>6}", metrics.batch_size);
  logger->info("  Input Length per Sample: {:>6} tokens",
               metrics.input_seq_length);
  logger->info("  Output Length per Sample: {:>6} tokens",
               metrics.output_seq_length);
  if (metrics.num_images > 0) {
    logger->info("  Number of Images: {:>6} images", metrics.num_images);
  }
  logger->info(
      "-----------------------------------------------------------------------"
      "-");
  // Brief Inference Performance
  logger->info(
      "-------------------- Inference Performance ---------------------");
  logger->info(
      "  Prefill  API Inference total Time: {:.2f} ms | Speed: {:.2f} "
      "tokens/s",
      results.metrics.prefill_perf_infos.infer_time,
      results.metrics.prefill_perf_infos.infer_speed);
  logger->info(
      "  Decode  API Inference total Time: {:.2f} ms | Speed: {:.2f} tokens/s",
      results.metrics.decode_perf_infos.infer_time,
      results.metrics.decode_perf_infos.infer_speed);
  logger->info(
      "  Vision  API Inference total Time: {:.2f} ms | Speed: {:.2f} images/s",
      results.metrics.vision_perf_infos.infer_time,
      results.metrics.vision_perf_infos.vision_infer_speed);

  // Summary metrics
  logger->info(
      "-----------------------------------------------------------------------"
      "-");
  logger->info(
      "-------------------- Overall Performance -----------------------");
  if (results.prefill_load_time > 0) {
    logger->info("  Prefill Model Load Time: {:>7.2f}ms",
                 results.prefill_load_time);
  }
  if (results.decode_load_time > 0) {
    logger->info("  Decode Model Load Time: {:>7.2f}ms",
                 results.decode_load_time);
  }
  if (results.vision_load_time > 0) {
    logger->info("  Vision Model Load Time: {:>7.2f}ms",
                 results.vision_load_time);
  }
  if (!post_init_dev_stats.empty()) {
    logger->info("  Model Load Device Memory:");
    for (const auto &[dev_id, device_stats] : post_init_dev_stats) {
      logger->info("    Device {} | Total: {} MB | Used: {} MB", dev_id,
                   device_stats.mem_info.mem_total,
                   device_stats.mem_info.mem_used);
    }
  }
  logger->info("  TTFT (Time To First Token): {:>7.2f} ms", metrics.ttft);
  logger->info("  TPOT (Time Per Output Token): {:>5.2f} ms/token",
               metrics.tpot);
  logger->info("  E2E Latency (End-to-End): {:>9.2f} seconds",
               metrics.e2e_time);
  logger->info("  E2E TPS (Throughput): {:>13.2f} tokens/s", metrics.e2e_tps);

  logger->info(
      "-----------------------------------------------------------------------"
      "-");
  // host info
#if defined(__linux__)
  logger->info(
      "-------------------- Memory Usage (Max Values) -----------------");
  logger->info("  Physical Memory: {}",
               formatMemorySize(host_mem_info.physical_memory));
  logger->info("  Virtual Memory: {}",
               formatMemorySize(host_mem_info.virtual_memory));
  logger->info("  Max Physical Memory: {}",
               formatMemorySize(max_host_mem_info.physical_memory));
  logger->info("  Max Virtual Memory: {}",
               formatMemorySize(max_host_mem_info.virtual_memory));
#endif
  // device info
  logger->info(
      "-----------------------------------------------------------------------"
      "-");
  logger->info(
      "-------------------- Device Stats (Max Values)  ----------------");
  for (const auto &[dev_id, device_stats] : end_device_stats) {
    logger->info("Device {}: ", dev_id);
    auto fmt = [](auto v, int prec, const char *unit) -> std::string {
      std::ostringstream o;
      o << std::fixed << std::setprecision(prec) << v << unit;
      return o.str();
    };
    logger->info("{:<15}|  {:<18} |  {:<18} |  {:<18} |", "Temperature",
                 fmt(device_stats.temperature_min, 2, "°C(Min)"),
                 fmt(device_stats.temperature_max, 2, "°C(Max)"),
                 fmt(device_stats.temperature_avg, 2, "°C(Avg)"));
    logger->info("{:<15}|  {:<18} |  {:<18} |  {:<18} |", "Power",
                 fmt(device_stats.power_min, 2, " W(Min)"),
                 fmt(device_stats.power_max, 2, " W(Max)"),
                 fmt(device_stats.power_avg, 2, " W(Avg)"));

    logger->info("{:<15}|  {:<18} |  {:<18} |  {:<18} |", "IPU Freq",
                 fmt(device_stats.ipu_freq_min, 2, " Mhz(Min)"),
                 fmt(device_stats.ipu_freq_max, 2, " Mhz(Max)"),
                 fmt(device_stats.ipu_freq_avg, 2, " Mhz(Avg)"));
    logger->info("{:<15}|  {:<18} |  {:<18} |  {:<18} |", "Mem Info",
                 fmt(device_stats.mem_info.mem_total, 2, " MB(Total)"),
                 fmt(device_stats.mem_info.mem_used, 2, " MB(Used)"),
                 fmt(device_stats.mem_info.mem_avail, 2, " MB(Avail)"));
    logger->info("{:<15}|  {:<18} |  {:<18} |  {:<18} |", "Mem Used",
                 fmt(device_stats.mem_used_min, 2, " MB(Min)"),
                 fmt(device_stats.mem_used_max, 2, " MB(Max)"),
                 fmt(format_device_memory_avg_as_int(device_stats.mem_used_avg),
                     0, " MB(Avg)"));
  }
  logger->info(
      "-----------------------------------------------------------------------"
      "-");

  // Vision stage performance (if any)
  if (metrics.num_images > 0 &&
      (metrics.vision_perf_infos.vision_total_time > 0 ||
       metrics.vision_perf_infos.vision_preprocess_time > 0)) {
    logger->info(
        "-------------------- Vision Stage Performance --------------------");
    logger->info("  Total Time: {:>7.2f}ms | Speed: {:>7.2f} images/s",
                 metrics.vision_perf_infos.vision_total_time,
                 metrics.vision_perf_infos.vision_total_speed);
    if (metrics.vision_perf_infos.vision_preprocess_time > 0) {
      logger->info(
          "  Preprocessing Time: {:>5.2f}ms | Speed: {:>7.2f} images/s",
          metrics.vision_perf_infos.vision_preprocess_time,
          metrics.vision_perf_infos.vision_preprocess_speed);
    } else {
      logger->info("  Preprocessing Time: Skipped (No operation)");
    }

    logger->info("  API SetInput total Time: {:>6.2f}ms",
                 metrics.vision_perf_infos.setinput_time);
    logger->info(
        "  API Inference total Time: {:>5.2f}ms | Speed: {:>7.2f} images/s",
        metrics.vision_perf_infos.infer_time,
        metrics.vision_perf_infos.vision_infer_speed);
    logger->info("  API GetOutput total Time: {:>5.2f}ms",
                 metrics.vision_perf_infos.getoutput_time);
    logger->info(
        "----------------------------------------------------------------------"
        "--");
  }

  // Prefill stage performance
  logger->info(
      "-------------------- Prefill Stage Performance -----------------");

  logger->info("  Total Time: {:>7.2f}ms | Speed: {:>7.2f} tokens/s",
               metrics.prefill_perf_infos.total_time,
               metrics.prefill_perf_infos.total_speed);
  if (metrics.prefill_perf_infos.tokenizer_time > 0) {
    logger->info("  Tokenization total Time: {:>7.2f}ms",
                 metrics.prefill_perf_infos.tokenizer_time);
  } else {
    logger->info("  Tokenization total Time: Skipped (No operation)");
  }
  logger->info("  Embedding total Time: {:>7.2f}ms",
               metrics.prefill_perf_infos.embedding_time);
  logger->info("  API SetInput total Time: {:>6.2f}ms",
               metrics.prefill_perf_infos.setinput_time);
  logger->info(
      "  API Inference total Time: {:>5.2f}ms | Speed: {:>7.2f} tokens/s",
      metrics.prefill_perf_infos.infer_time,
      metrics.prefill_perf_infos.infer_speed);
  logger->info("  API GetOutput total Time: {:>5.2f}ms",
               metrics.prefill_perf_infos.getoutput_time);
  logger->info(
      "-----------------------------------------------------------------------"
      "-");

  // Decode stage performance
  logger->info(
      "-------------------- Decode Stage Performance ------------------");
  logger->info("  Total Time: {:>7.2f}ms | Speed: {:>7.2f} tokens/s",
               metrics.decode_perf_infos.total_time,
               metrics.decode_perf_infos.total_speed);
  if (metrics.decode_perf_infos.tokenizer_time > 0) {
    logger->info("  Tokenization total Time: {:>7.2f}ms",
                 metrics.decode_perf_infos.tokenizer_time);
  } else {
    logger->info("  Tokenization total Time: Skipped (No operation)");
  }
  logger->info("  Embedding total Time: {:>7.2f}ms",
               metrics.decode_perf_infos.embedding_time);
  logger->info("  API SetInput avg Time: {:>6.2f}ms/token",
               metrics.decode_perf_infos.setinput_time_per_token);
  logger->info(
      "  API Inference avg Time: {:>5.2f}ms/token | Speed: {:>7.2f} tokens/s",
      metrics.decode_perf_infos.infer_time_per_token,
      metrics.decode_perf_infos.infer_speed);
  logger->info("  API GetOutput avg Time: {:>5.2f}ms/token",
               metrics.decode_perf_infos.getoutput_time_per_token);

  logger->info(
      "======================================================================="
      "=\n\n\n");
}

void PerfDumper::generateYamlFile() {
  if (!dump_file.empty()) {
    try {
      fs::path file_path(dump_file);
      fs::path parent_dir = file_path.parent_path();

      if (!parent_dir.empty() && !fs::exists(parent_dir)) {
        fs::create_directories(parent_dir);
        std::cout << "Successfully created directory: " << parent_dir
                  << std::endl;
      }

      std::ofstream out(dump_file);
      if (!out.is_open()) {
        std::cerr << "Error: Cannot open file " << dump_file << std::endl;
        return;
      }

      // Serialize the YAML node to a string and write to file
      out << this->root;
      out.close();

      std::cout << COLOR_GREEN
                << "Successfully wrote Perf Result to YAML file: " << dump_file
                << COLOR_RESET << std::endl;
    } catch (const fs::filesystem_error &e) {
      std::cerr << "Error: Failed to create directory - " << e.what()
                << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error: Failed to write YAML file - " << e.what()
                << std::endl;
    }
  }
  std::cout << COLOR_GREEN
            << "Every Loop Performance Result has been written to " << log_file
            << COLOR_RESET << std::endl;
}