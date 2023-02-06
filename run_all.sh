#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname ${BASH_SOURCE[0]})"; pwd)

${SCRIPT_PATH}/Apollo/run_all.sh
