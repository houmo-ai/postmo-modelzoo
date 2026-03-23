#!/usr/bin/env bash
set -e

houmo_target="${HOUMO_TARGET}"
if [[ -z "$houmo_target" || "$houmo_target" != "xh2" ]]; then
    echo "Only supports HOUMO_TARGET as xh2." >&2
    exit 1
fi

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${SCRIPT_DIR}" || exit 1

arch=$(uname -m)
if [[ "$arch" == "aarch64" ]]; then
    export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
fi

echo "Download qwen3-8b-16k 4batch model"
python3 get_model.py

if [[ ! -f "output/${houmo_target}/qwen3_prefill.hmm" ]]; then
    echo "Missing file: qwen3_prefill.hmm" >&2
    exit 1
fi

if [[ ! -f "output/${houmo_target}/qwen3_decode.hmm" ]]; then
    echo "Missing file: qwen3_decode.hmm" >&2
    exit 1
fi

if [[ ! -f "output/${houmo_target}/hmquant/quant_embedding.pt" ]]; then
    echo "Missing file: output/${houmo_target}/hmquant/quant_embedding.pt" >&2
    exit 1
fi

echo "Run qwen3-8b-16k 4batch demo"
python3 demo_multibatch.py --forbid_flush