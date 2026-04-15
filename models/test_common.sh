#!/usr/bin/env bash

if [[ -n "${IMODELZOO_TEST_COMMON_SH:-}" ]]; then
    return 0
fi
IMODELZOO_TEST_COMMON_SH=1

show_help() {
    echo "Usage: $0 [options]"
    echo "  -s, --step              execution step, default is demo, support: demo, build, quant, all."
    echo "  -skip, --skip_download  skip get_model"
    echo "  -m, --model_size        Model size, default is null"
    echo "  -n, --model_name        Model name, default is null"
    echo "  -h, --help              help infomation"
    exit 0
}

parse_args() {
    STEP="${STEP:-demo}"
    SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-false}"
    MODEL_SIZE="${MODEL_SIZE:-}"
    MODEL_NAME="${MODEL_NAME:-}"

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help)
                show_help
                ;;
            -s|--step)
                if [[ $# -lt 2 ]]; then
                    echo "Error: Missing value for parameter '$1'" >&2
                    show_help
                fi
                STEP="$2"
                shift 2
                ;;
            -skip|--skip_download)
                SKIP_DOWNLOAD="true"
                shift
                ;;
            -m|--model_size)
                if [[ $# -lt 2 ]]; then
                    echo "Error: Missing value for parameter '$1'" >&2
                    show_help
                fi
                MODEL_SIZE="$2"
                shift 2
                ;;
            -n|--model_name)
                if [[ $# -lt 2 ]]; then
                    echo "Error: Missing value for parameter '$1'" >&2
                    show_help
                fi
                MODEL_NAME="$2"
                shift 2
                ;;
            *)
                echo "Error: Unknown parameter '$1'" >&2
                show_help
                ;;
        esac
    done

    case "${STEP}" in
        all|demo|build|quant)
            ;;
        *)
            echo "Error: Unsupported step '${STEP}', support: all, demo, build, quant." >&2
            show_help
            ;;
    esac
}

should_run_step() {
    local step="$1"

    [[ "${STEP:-all}" == "all" || "${STEP:-all}" == "${step}" ]]
}

should_skip_download() {
    [[ "${SKIP_DOWNLOAD:-false}" == "true" ]]
}

setup_python_venv() {
    local venv_dir="$1"
    local requirements_file="${2:-requirements.txt}"
    local venv_label="${3:-${venv_dir} demo}"
    local python_exe=
    local system_site_packages=
    local venv_python=
    local venv_site_packages=

    TEST_VENV_ACTIVE=0
    TEST_VENV_DIR="${venv_dir}"

    if [[ ! -f "${requirements_file}" ]]; then
        return 1
    fi

    python_exe=$(command -v python3)
    if [[ -z "${python_exe}" ]]; then
        echo "Error: python3 not found." >&2
        return 1
    fi

    echo "⚠ Create python3 venv for ${venv_label}."
    system_site_packages=$("${python_exe}" -c "import site; print(site.getsitepackages()[0])")
    if [[ "${python_exe}" == */opt/venv* ]]; then
        virtualenv --python="${python_exe}" --extra-search-dir="${system_site_packages}" "${venv_dir}"
        venv_python="${venv_dir}/bin/python3"
        venv_site_packages=$("${venv_python}" -c "import site; print(site.getsitepackages()[0])")
        {
            echo 'export ORIGINAL_PYTHONPATH=$PYTHONPATH'
            echo "export PYTHONPATH=${venv_site_packages}:${system_site_packages}:\$ORIGINAL_PYTHONPATH"
        } >> "${venv_dir}/bin/activate"
        {
            echo 'export PYTHONPATH=$ORIGINAL_PYTHONPATH'
            echo 'unset ORIGINAL_PYTHONPATH'
        } >> "${venv_dir}/bin/deactivate"
        sed -i 's/include-system-site-packages = true/include-system-site-packages = false/g' "${venv_dir}/pyvenv.cfg"
    else
        virtualenv --python="${python_exe}" --system-site-packages "${venv_dir}"
        venv_python="${venv_dir}/bin/python3"
    fi

    source "${venv_dir}/bin/activate"
    pip3 install -r "${requirements_file}"

    TEST_VENV_ACTIVE=1
    TEST_VENV_PYTHON="${venv_python}"
    TEST_VENV_SITE_PACKAGES=$("${venv_python}" -c "import site; print(site.getsitepackages()[0])")
    return 0
}

cleanup_python_venv() {
    local venv_dir="${1:-${TEST_VENV_DIR:-}}"

    if [[ "${TEST_VENV_ACTIVE:-0}" -eq 1 ]] && declare -F deactivate >/dev/null 2>&1; then
        deactivate
    fi
    if [[ -n "${venv_dir}" && -d "${venv_dir}" ]]; then
        rm -rf "${venv_dir}"
    fi

    TEST_VENV_ACTIVE=0
}

check_python_package() {
    local package_pattern="$1"

    echo "================================"
    echo "Checking python3 package: ${package_pattern}"
    if ! command -v python3 &>/dev/null || ! command -v pip3 &>/dev/null; then
        echo "⚠ Not found python3 or pip3."
        return 2
    fi

    if pip3 list --format=columns 2>/dev/null | grep -E "^${package_pattern}" >/dev/null 2>&1; then
        echo "✓ Found python3 package: ${package_pattern}"
        pip3 list --format=columns 2>/dev/null | grep -E "^${package_pattern}" | while read -r line; do
            echo "  - ${line}"
        done
        return 0
    fi

    echo "✗ Not found package: ${package_pattern}"
    return 1
}

check_gpu() {
    local mode="${1:-warn}"
    local gpu_count=0

    echo "Checking GPU..."
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        if [[ "${mode}" == "require" ]]; then
            echo "Error: NVIDIA driver (nvidia-smi) not found; GPU is required." >&2
        else
            echo "⚠ Not install NVIDIA GPU driver."
        fi
        return 1
    fi

    gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader,nounits 2>/dev/null | wc -l)
    if [[ "${gpu_count}" -gt 0 ]]; then
        TEST_GPU_COUNT="${gpu_count}"
        echo "Found NVIDIA GPU, count: ${gpu_count}"
        return 0
    fi

    if [[ "${mode}" == "require" ]]; then
        echo "Error: No NVIDIA GPU device detected; GPU is required." >&2
    else
        echo "⚠ Not found NVIDIA GPU device."
    fi
    return 1
}