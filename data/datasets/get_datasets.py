# Copyright 2025 HOUMO AI
#
# File: get_datasets.py
# Description:
#   Download datasets used by internal tests.
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

import argparse
import sys
from pathlib import Path

from hmatc.utils.utils import get_file_from_jfrog

RESOURCE_PATH = "../toolchain/support/opensource_datasets.zip"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download and extract datasets used by internal tests."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory used to save and extract the archive (default: script directory).",
    )
    parser.add_argument(
        "--remove-archive",
        action="store_true",
        help="Remove the downloaded archive after successful extraction.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    save_path = get_file_from_jfrog(
        RESOURCE_PATH, str(output_dir), str(output_dir)
    )
    if not save_path:
        print(f"Failed to download or extract {RESOURCE_PATH}.", file=sys.stderr)
        return 1

    archive_path = Path(save_path)
    if args.remove_archive:
        archive_path.unlink(missing_ok=True)

    print(f"Datasets extracted to: {output_dir}")
    if not args.remove_archive:
        print(f"Downloaded archive: {archive_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
