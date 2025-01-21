#!/usr/bin/env bash

# main path
__dir="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
export HOUMO_EXAMPLES_PATH=${__dir}

# common define
PRINT_GREEN() { echo -e "\033[1;32m$@\033[0m"; }
PRINT_YELLOW() { echo -e "\033[1;33m$@\033[0m"; }

PRINT_GREEN "HOUMO_EXAMPLES_PATH is $HOUMO_EXAMPLES_PATH"
