#!/bin/bash

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

python3 bandwidth_perf.py --type r "$@"
sleep 5
python3 bandwidth_perf.py --type w "$@"