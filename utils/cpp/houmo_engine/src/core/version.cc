/*
 * Copyright (c) 2026 HOUMO AI
 *
 * File: version.cc
 * Description:
 *   Houmo Inference Framework version information.
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

#include "base/houmo.h"

namespace houmo {

std::string version() {
    return "0.1.0";
}

std::string build_info() {
    return "Houmo Inference Framework v0.1.0\n"
           "Build: " __DATE__ " " __TIME__ "\n"
           "Compiler: "
#if defined(__GNUC__)
           "GCC " __VERSION__
#elif defined(__clang__)
           "Clang " __clang_version__
#elif defined(_MSC_VER)
           "MSVC "
#endif
           ;
}

} // namespace houmo
