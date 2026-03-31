#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Get version from environment or use default
HOUMO_VERSION="${HOUMO_VERSION:-1.1.0}"

# Check if already in a venv, create sub-venv to isolate dependencies
VENV_DIR="qwen3_vl_demo"
VENV_FLAG=0

if [[ $PY_EXE == */opt/venv* ]] || [[ -n "$VIRTUAL_ENV" ]]; then
    VENV_FLAG=1
    echo "Detected venv environment, creating sub-venv for ${VENV_DIR} demo..."
    PY_EXE=$(command -v python3)
    SITE_PACKAGES=$($PY_EXE -c "import site; print(site.getsitepackages()[0])")

    # Use extra-search-dir to inherit parent packages
    virtualenv --python=$PY_EXE --extra-search-dir=$SITE_PACKAGES $VENV_DIR
    VENV_PYTHON="${VENV_DIR}/bin/python3"
    VENV_SITE=$(${VENV_PYTHON} -c "import site; print(site.getsitepackages()[0])")
    echo "export ORIGINAL_PYTHONPATH=\$PYTHONPATH" >> $VENV_DIR/bin/activate
    echo "export PYTHONPATH=${VENV_SITE}:${SITE_PACKAGES}:\$ORIGINAL_PYTHONPATH" >> $VENV_DIR/bin/activate
    echo "export PYTHONPATH=\$ORIGINAL_PYTHONPATH" >> $VENV_DIR/bin/deactivate
    echo "unset ORIGINAL_PYTHONPATH" >> $VENV_DIR/bin/deactivate
    sed -i 's/include-system-site-packages = true/include-system-site-packages = false/g' $VENV_DIR/pyvenv.cfg

    source $VENV_DIR/bin/activate
fi

# Download hmm model
echo "Downloading hmm model..."
python get_model.py --download-dir ./models

# Install specific dependencies
pip3 install transformers==4.57.1 qwen_vl_utils -i https://pypi.tuna.tsinghua.edu.cn/simple

# Run evaluation
echo "Running evaluation..."
hmeval \
    --model ./hm_xh2_qwen3_vl.py \
    --model-dir ./models/hmm_xh2_qwen3-vl_4b_256_32k_b1_1chip_2cores_v${HOUMO_VERSION} \
    --dataset mm_bench \
    --limit 1 \
    --model-args tokenizer_dir=./models/tokenizers

# Deactivate and cleanup sub-venv if created
if [[ "$VENV_FLAG" -eq 1 ]]; then
    deactivate
    rm -rf $VENV_DIR
fi

echo "Done!"