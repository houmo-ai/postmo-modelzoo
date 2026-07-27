/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: tcim_utils.h
 * Description:
 *   TCIM Runtime utility functions for error checking and status handling.
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

#pragma once

#include <iostream>
#include <sstream>
#include <stdexcept>

#include "tcim/tcim_runtime.h"

namespace houmo {

/**
 * @brief Check TCIM Runtime return status
 * @param status Status code returned by TCIM API
 * @param file Source file name of the call site
 * @param line Line number of the call site
 * @throws std::runtime_error if status is not OK
 */
inline void CheckTcimRetStatus(const tcim::Status& status, const char* file,
                               int line) {
  if (status != tcim::Status::OK) {
    std::ostringstream err_msg;
    err_msg << "TCIM Runtime error: " << static_cast<int>(status)
            << " at " << file << ":" << line;
    throw std::runtime_error(err_msg.str());
  }
}

}  // namespace houmo

/**
 * @brief Macro to check TCIM return status (automatically records file and line)
 */
#define CHECK_TCIM_RET_STATUS(status) \
  houmo::CheckTcimRetStatus(status, __FILE__, __LINE__)
