#!/usr/bin/python3
# -*- coding: utf-8 -*-
# Copyright 2025 HOUMO AI
#
# File: model.py
# Description:
#   Package, compress, and upload the compiled model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
import os
from datetime import datetime
from hmatc.utils import logger
from hmatc.utils.utils import *

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
logger.info("Compressing hmmodel...")
hmcc_version = get_package_version(f"houmo-tcim-xh2")
runtime_version = get_package_version(f"houmo_tcim_runtime_xh2")
with open(os.path.join("output", "xh2", "VERSION.txt"), "w") as f:
    f.write(f"hmquant_version: {get_hmquant_xh2_version()}\n")
    f.write(f"tcim_version: {hmcc_version}\n")
    f.write(f"tcim_runtime_version: {runtime_version}\n")
    f.write(f"build_time: {now}\n")
filename = f"ppocrv3_det_xh2_b1_1core_O2_{get_houmo_version()}.tar.xz"
compress_hmm_path = os.path.join(
    "output",
    "xh2",
    filename,
)
hmm_path = os.path.join("output", "xh2/ppocrv3_det_xh2_b1_1roi_1core_O2_static.hmm")
compress_files_to_tar_xz_with_progress(
    [hmm_path, os.path.join("output", "xh2", "VERSION.txt")],
    compress_hmm_path,
)
logger.info(
    f"MD5: {get_file_md5(compress_hmm_path)}, save path: {compress_hmm_path}"
)
upload_file_to_artifactory(
    compress_hmm_path,
    f"models/xh2-{get_houmo_version()}/ppocrv3_det/{filename}",
    max_retries=3,
)
logger.info(f"Compressing hmmodel done.")
