/*
 * Copyright (c) 2025 HOUMO AI
 *
 * File: PerfDumper.cc
 * Description:
 *   PerfDumper Implementation - dump perf metrics to json file
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
#include "perf_dumper.h"

PerfDumper::PerfDumper() { dump_file = ""; }

PerfDumper::~PerfDumper() { perf_metrics = write_json::array(); }

void PerfDumper::setJsonFile(const std::string &json_file, bool run_json_perf) {
  if (dump_file.empty()) {
    dump_file = json_file;
  } else {
    if (!run_json_perf) {
      dump_file = json_file;
    }
  }
}

void PerfDumper::dumpPerf(const PerfSettings &perf_settings,
                          const InferenceMetricsWithLoadTime &results) {
  write_json model_metrics;
  // perf settings
  model_metrics["ModelName"] = perf_settings.model_name;
  model_metrics["prefill"] = perf_settings.prefill_path;
  model_metrics["decode"] = perf_settings.decode_path;
  model_metrics["visual"] = perf_settings.visual_path;
  model_metrics["embedding"] = perf_settings.embedding_path;
  model_metrics["input"] = perf_settings.input_tokens_len;
  model_metrics["stop"] = perf_settings.stop_tokens_len;
  model_metrics["ndevices"] = perf_settings.ndevices;
  model_metrics["batch"] = perf_settings.batch_size;
  model_metrics["loop"] = perf_settings.loop_count;
  model_metrics["LazyMode"] = perf_settings.LazyMode;
  model_metrics["warm_up"] = perf_settings.warm_up;

  // perf results
  model_metrics["input_token"] = results.metrics.input_seq_length;
  model_metrics["output_token"] = results.metrics.output_seq_length;
  model_metrics["prefill_load_time"] =
      round_to_3_decimals(results.prefill_load_time);
  model_metrics["decode_load_time"] =
      round_to_3_decimals(results.decode_load_time);
  model_metrics["vision_load_time"] =
      round_to_3_decimals(results.vision_load_time);
  model_metrics["prefill_time"] =
      round_to_3_decimals(results.metrics.prefill_perf_infos.total_time);
  model_metrics["decode_time"] =
      round_to_3_decimals(results.metrics.decode_perf_infos.total_time);
  model_metrics["vision_time"] =
      round_to_3_decimals(results.metrics.vision_perf_infos.total_time);
  model_metrics["prefill_speed"] =
      round_to_3_decimals(results.metrics.prefill_perf_infos.total_speed);
  model_metrics["decode_speed"] =
      round_to_3_decimals(results.metrics.decode_perf_infos.total_speed);
  model_metrics["vision_speed"] =
      round_to_3_decimals(results.metrics.vision_perf_infos.total_speed);
  model_metrics["prefill_infer_time_avg"] =
      round_to_3_decimals(results.metrics.prefill_perf_infos.infer_time);
  model_metrics["decode_infer_time_avg"] = round_to_3_decimals(
      results.metrics.decode_perf_infos.infer_time_per_token);
  model_metrics["vision_infer_time_avg"] =
      round_to_3_decimals(results.metrics.vision_perf_infos.infer_time);
  model_metrics["prefill_infer_speed_avg"] =
      round_to_3_decimals(results.metrics.prefill_perf_infos.infer_speed);
  model_metrics["decode_infer_speed_avg"] =
      round_to_3_decimals(results.metrics.decode_perf_infos.infer_speed);
  model_metrics["vision_infer_speed_avg"] =
      round_to_3_decimals(results.metrics.vision_perf_infos.infer_speed);
  model_metrics["prefill_embedding_time"] =
      round_to_3_decimals(results.metrics.prefill_perf_infos.embedding_time);
  model_metrics["decode_embedding_time"] =
      round_to_3_decimals(results.metrics.decode_perf_infos.embedding_time);
  model_metrics["TTFT"] = round_to_3_decimals(results.metrics.ttft);
  model_metrics["TPOT"] = round_to_3_decimals(results.metrics.tpot);
  model_metrics["e2e_latency"] = round_to_3_decimals(results.metrics.e2e_time);
  model_metrics["e2e_tps"] = round_to_3_decimals(results.metrics.e2e_tps);

  perf_metrics.emplace_back(model_metrics);
}

void PerfDumper::generateJsonFile() {
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
      root["Perf_Metrics"] = perf_metrics;
      out << root.dump(4);
      out.close();
      std::cout << "Successfully wrote JSON file: " << dump_file << std::endl;
    } catch (const fs::filesystem_error &e) {
      std::cerr << "Error: Failed to create directory - " << e.what()
                << std::endl;
    } catch (const std::exception &e) {
      std::cerr << "Error: Failed to write JSON file - " << e.what()
                << std::endl;
    }
  }
}
