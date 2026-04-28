#!/usr/bin/env bash

if [[ -n "${IMODELZOO_TEST_COMMON_SH:-}" ]]; then
    return 0
fi
IMODELZOO_TEST_COMMON_SH=1
declare -Ag IMODELZOO_PYTHON_PACKAGE_CHECK_CACHE=()

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all."
    echo "                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build."
    echo "  -b, --multi_batch       Enable multi-batch mode."
    echo "  -skip, --skip_download  Skip model download steps."
    echo "  -size, --model_size     Model size."
    echo "  -name, --model_name     Model name."
    echo "  -h, --help              Show this help message."
    exit 0
}

parse_args() {
    local step_explicit=0
    local normalized_steps=""
    local raw_step=

    STEP="${STEP:-demo}"
    MULTI_BATCH="${MULTI_BATCH:-false}"
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
                if [[ "${step_explicit}" -eq 0 ]]; then
                    STEP=""
                    step_explicit=1
                fi
                if [[ -n "${STEP}" ]]; then
                    STEP+=",$2"
                else
                    STEP="$2"
                fi
                shift 2
                ;;
            -b|--multi_batch)
                MULTI_BATCH="true"
                shift
                ;;
            -skip|--skip_download)
                SKIP_DOWNLOAD="true"
                shift
                ;;
            -size|--model_size)
                if [[ $# -lt 2 ]]; then
                    echo "Error: Missing value for parameter '$1'" >&2
                    show_help
                fi
                MODEL_SIZE="$2"
                shift 2
                ;;
            -name|--model_name)
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

    for raw_step in ${STEP//,/ }; do
        case "${raw_step}" in
            all)
                STEP="all"
                return 0
                ;;
            demo|build|quant)
                case ",${normalized_steps}," in
                    *,${raw_step},*)
                        ;;
                    *)
                        if [[ -n "${normalized_steps}" ]]; then
                            normalized_steps+=",${raw_step}"
                        else
                            normalized_steps="${raw_step}"
                        fi
                        ;;
                esac
                ;;
            *)
                echo "Error: Unsupported step '${raw_step}', support: all, demo, build, quant." >&2
                show_help
                ;;
        esac
    done

    STEP="${normalized_steps:-demo}"
}

should_run_step() {
    local step="$1"
    local selected_step=

    if [[ "${STEP:-all}" == "all" ]]; then
        return 0
    fi

    for selected_step in ${STEP//,/ }; do
        if [[ "${selected_step}" == "${step}" ]]; then
            return 0
        fi
    done

    return 1
}

should_skip_download() {
    [[ "${SKIP_DOWNLOAD:-false}" == "true" ]]
}

check_houmo_target() {
    local expected_target="$1"
    local current_target="${HOUMO_TARGET:-}"

    if [[ -z "${expected_target}" ]]; then
        echo "Error: Missing expected HOUMO_TARGET value." >&2
        return 1
    fi

    if [[ -z "${current_target}" || "${current_target}" != "${expected_target}" ]]; then
        echo "Only supports HOUMO_TARGET as ${expected_target}." >&2
        return 1
    fi

    return 0
}

link_subdirs() {
    if [[ $# -ne 2 ]]; then
        echo "link_subdirs <source_dir> <target_dir>"
        return 1
    fi

    local src_dir="$1"
    local dest_dir="$2"

    if [[ ! -d "${src_dir}" ]]; then
        echo "error: ${src_dir} unavailable!"
        return 1
    fi

    mkdir -p "${dest_dir}"

    for subdir in "${src_dir}"/*/; do
        [[ -d "${subdir}" ]] || continue
        local subdir_name
        local link_path
        local abs_subdir

        subdir_name=$(basename "${subdir}")
        link_path="${dest_dir}/${subdir_name}"
        if [[ -e "${link_path}" ]]; then
            rm -rf "${link_path}"
        fi

        abs_subdir=$(realpath "${subdir}")
        ln -s "${abs_subdir}" "${link_path}"
    done
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
            echo "export PYTHONPATH=${venv_site_packages}:\$ORIGINAL_PYTHONPATH:${system_site_packages}"
        } >> "${venv_dir}/bin/activate"
        {
            echo 'export PYTHONPATH=$ORIGINAL_PYTHONPATH'
            echo 'unset ORIGINAL_PYTHONPATH'
        } >> "${venv_dir}/bin/deactivate"
        sed -i 's/include-system-site-packages = true/include-system-site-packages = false/g' "${venv_dir}/pyvenv.cfg"

        if [[ ! -d "$venv_dir/lib/python3.12/site-packages/distutils" ]]; then
            ln -s ${system_site_packages}/setuptools/_distutils \
                $venv_dir/lib/python3.12/site-packages/distutils
        fi
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

    if [[ -v "IMODELZOO_PYTHON_PACKAGE_CHECK_CACHE[${package_pattern}]" ]]; then
        return "${IMODELZOO_PYTHON_PACKAGE_CHECK_CACHE[${package_pattern}]}"
    fi

    echo "================================"
    echo "Checking python3 package: ${package_pattern}"
    if ! command -v python3 &>/dev/null || ! command -v pip3 &>/dev/null; then
        echo "⚠ Not found python3 or pip3."
        IMODELZOO_PYTHON_PACKAGE_CHECK_CACHE["${package_pattern}"]=2
        return 2
    fi

    if pip3 list --format=columns 2>/dev/null | grep -E "^${package_pattern}" >/dev/null 2>&1; then
        echo "✓ Found python3 package: ${package_pattern}"
        pip3 list --format=columns 2>/dev/null | grep -E "^${package_pattern}" | while read -r line; do
            echo "  - ${line}"
        done
        IMODELZOO_PYTHON_PACKAGE_CHECK_CACHE["${package_pattern}"]=0
        return 0
    fi

    echo "✗ Not found package: ${package_pattern}"
    IMODELZOO_PYTHON_PACKAGE_CHECK_CACHE["${package_pattern}"]=1
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

    gpu_count=$(nvidia-smi -L 2>/dev/null | wc -l)
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

require_python_package() {
    local package_pattern="$1"
    local step_name="$2"
    local package_status=0

    if [[ -z "${package_pattern}" || -z "${step_name}" ]]; then
        echo "Error: Missing package pattern or step name for package validation." >&2
        return 1
    fi

    if check_python_package "${package_pattern}"; then
        return 0
    fi

    package_status=$?
    if [[ "${package_status}" -eq 2 ]]; then
        echo "Error: ${step_name} requires python3, pip3, and package '${package_pattern}'." >&2
    else
        echo "Error: ${step_name} requires python package '${package_pattern}'." >&2
    fi
    return 1
}

check_step_python_packages() {
    if should_run_step "quant"; then
        require_python_package "hmquant" "quant" || return 1
    fi

    if should_run_step "build"; then
        require_python_package "houmo-tcim" "build" || return 1
    fi

    if should_run_step "demo"; then
        require_python_package "houmo_tcim_runtime" "demo" || return 1
    fi

    return 0
}
