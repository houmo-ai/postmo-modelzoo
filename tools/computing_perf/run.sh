#!/bin/bash

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

if [ "${HDPL_PLATFORM:-}" = "ISIM" ]; then
    echo "ISIM platform does not support this tool."
    exit 0
fi

python3 computing_perf.py
