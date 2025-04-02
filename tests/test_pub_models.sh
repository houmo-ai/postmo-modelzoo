#!/usr/bin/env bash

set -e

SCRIPT_PATH=$(cd "$(dirname "${BASH_SOURCE[0]}")"; pwd)

cd "${SCRIPT_PATH}"


bash pub_models/test_adr_models.sh
bash pub_models/test_cv_models.sh
bash pub_models/test_llm_models.sh
