#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
MODELS_DIR="${SCRIPT_DIR}"
while [[ ! -f "${MODELS_DIR}/test_common.sh" && "${MODELS_DIR}" != "/" ]]; do
    MODELS_DIR="$(dirname "${MODELS_DIR}")"
done
if [[ ! -f "${MODELS_DIR}/test_common.sh" ]]; then
    echo "Error: test_common.sh not found." >&2
    exit 1
fi
source "${MODELS_DIR}/test_common.sh"

COMMON_PARSE_ARGS_DEFINITION="$(declare -f parse_args)"
eval "${COMMON_PARSE_ARGS_DEFINITION/parse_args/common_parse_args}"
unset COMMON_PARSE_ARGS_DEFINITION

STEP="demo"
CONFIG_PATH="./configs/gemma4_e2b.yml"
MODEL_PATH=""
SKIP_DOWNLOAD="false"
SYSTEM_PROMPT_ARGS=()

show_help() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -s, --step              Step to run. Default: demo. Choices: demo, build, quant, all."
    echo "                          Supports comma-separated values or repeated flags, e.g. -s quant,build or -s quant -s build."
    echo "  -c, --config            Configuration file path relative to the Gemma4 directory."
    echo "                          Default: ./configs/gemma4_e2b.yml."
    echo "  --model                 Model directory passed to demo.py."
    echo "  --ndevice               Override the download device count and set the maximum devices for demo.py."
    echo "  --context_length        Override the context length used by get_model.py."
    echo "  --prefill_length        Override the prefill length used by get_model.py."
    echo "  --skip_download         If specified, all dependencies must have been fully downloaded previously without this flag. Already downloaded dependencies won't be re-downloaded regardless of this parameter."
    echo "  --system_prompt         System prompt passed through to demos that support it."
    echo "  -h, --help              Show this help message."
    exit 0
}

parse_args() {
    local common_args=()

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -c|--config)
                if [[ $# -lt 2 ]]; then
                    echo "Error: Missing value for parameter '$1'" >&2
                    show_help
                fi
                CONFIG_PATH="$2"
                shift 2
                ;;
            --model)
                if [[ $# -lt 2 ]]; then
                    echo "Error: Missing value for parameter '$1'" >&2
                    show_help
                fi
                MODEL_PATH="$2"
                shift 2
                ;;
            *)
                common_args+=("$1")
                shift
                ;;
        esac
    done

    common_parse_args "${common_args[@]}"
}

parse_args "$@"

run_get_model() {
    local file_type="$1"
    local get_model_args=(--config "${CONFIG_PATH}" --type "${file_type}")

    if [[ -n "${MODEL_NAME:-}" ]]; then
        get_model_args+=(--model_name "${MODEL_NAME}")
    fi
    if [[ -n "${MODEL_SIZE:-}" ]]; then
        get_model_args+=(--model_size "${MODEL_SIZE}")
    fi
    if [[ -n "${BATCH:-}" ]]; then
        get_model_args+=(--batch "${BATCH}")
    fi
    if [[ -n "${NDEVICE:-}" ]]; then
        get_model_args+=(--ndevice "${NDEVICE}")
    fi
    if [[ -n "${CONTEXT_LENGTH:-}" ]]; then
        get_model_args+=(--context_length "${CONTEXT_LENGTH}")
    fi
    if [[ -n "${PREFILL_LENGTH:-}" ]]; then
        get_model_args+=(--prefill_length "${PREFILL_LENGTH}")
    fi

    python3 get_model.py "${get_model_args[@]}"
}

cd "${SCRIPT_DIR}"

if [[ ! -f "${CONFIG_PATH}" ]]; then
    echo "Error: Config file not found: ${CONFIG_PATH}" >&2
    exit 1
fi

TEST_VENV_ACTIVE=0
dir_path="gemma4_venv"
trap 'if [[ "${TEST_VENV_ACTIVE:-0}" -eq 1 ]]; then cleanup_python_venv "${dir_path}"; fi' EXIT

if [[ -f "${SCRIPT_DIR}/requirements.txt" ]]; then
    export PYTHONPATH="${PYTHONPATH:-}"
    setup_python_venv "${dir_path}" "${SCRIPT_DIR}/requirements.txt" "Gemma4 demo"

    gptq_requirements="${HOUMO_EXAMPLES_PATH:-}/hmodel/gptqmodel/requirements.txt"
    if [[ -n "${HOUMO_EXAMPLES_PATH:-}" && -f "${gptq_requirements}" ]]; then
        python3 -m pip install -r "${gptq_requirements}"
    fi
fi

check_step_python_packages || exit 1

if should_run_step "quant"; then
    if ! should_skip_download; then
        echo "Download raw model."
        run_get_model raw
    fi
    echo "Start model quantization."
    hmatc quant -c "${CONFIG_PATH}"
fi

if should_run_step "build"; then
    echo "Start model compilation."
    hmatc build -c "${CONFIG_PATH}"
fi

if should_run_step "demo"; then
    if [[ -z "${MODEL_PATH}" && "${CONFIG_PATH}" != "./configs/gemma4_e2b.yml" ]]; then
        echo "Warning: --model is not specified; demo.py will use its default e2b model directory." >&2
    fi

    if [[ "${STEP}" == "demo" ]] && ! should_skip_download; then
        echo "Download pre-compiled model."
        run_get_model hmm
    fi

    echo "Execute python demo."
    demo_args=()
    if [[ -n "${MODEL_PATH}" ]]; then
        demo_args+=(--model "${MODEL_PATH}")
    fi
    if [[ -n "${NDEVICE:-}" ]]; then
        demo_args+=(--ndevice "${NDEVICE}")
    fi
    demo_args+=("${SYSTEM_PROMPT_ARGS[@]}")
    python3 demo.py "${demo_args[@]}"
fi
