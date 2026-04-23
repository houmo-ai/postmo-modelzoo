
/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: tcim_runtime_utils.h
 * Description:
 *   Check TCIM Runtime API ret Code and Debug Model Input Datas.
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
#ifndef __TCIM_RUNTIME_UTILS_H__
#define __TCIM_RUNTIME_UTILS_H__

#include <sstream>

#include "tcim/tcim_runtime.h"
inline void CheckTcimRetStatus(const tcim::Status &status,
                               const char *file = __FILE__,
                               int line = __LINE__) {
  if (status != tcim::Status::OK) {
    std::ostringstream err_msg;
    err_msg << "tcim_runtime ret Status is not OK! "
            << "File: " << file << ", Line: " << line
            << ", Current ret Status: " << static_cast<int>(status);

    throw std::runtime_error(err_msg.str());
  }
}

#define CHECK_TCIM_RET_STATUS(status) \
  CheckTcimRetStatus(status, __FILE__, __LINE__)

#endif  // __TCIM_RUNTIME_UTILS_H__