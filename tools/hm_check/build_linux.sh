#!/usr/bin/env bash
# build_linux.sh - build hm-check on Linux using CMake
# Usage:
#   ./build_linux.sh -b build -c Release -j 8 --tcim /opt/tcim --houmo /opt/houmo --install
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SOURCE_DIR}/build"
CONFIG=Release
GENERATOR=""
JOBS=$(nproc 2>/dev/null || echo 4)
TCIM=""
HOUMO=""
INSTALL=1
INSTALL_DIR=""  # Initialize INSTALL_DIR variable

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -s DIR            Source directory (default: script directory)
  -b DIR            Build directory (default: ./build)
  -c CONFIG         Build configuration: Release or Debug (default: Release)
  -g GENERATOR      CMake generator (optional, e.g. "Ninja")
  -j JOBS           Parallel jobs for build (default: number of CPUs)
  --tcim PATH       Set TCIM_RUNTIME_PATH (default: read from TCIM_RUNTIME_PATH env var)
  --houmo PATH      Set HOUMO_SDK_PATH (default: read from HOUMO_SDK_PATH env var)
  --install [PATH]  Install after building (default: ${SOURCE_DIR}/../bin, always enabled by default)
  --no-install      Skip installation after building
  -h                Show this help

Example:
  $0 -b build -c Release -j 8 --tcim $DADAO_VENV/lib/python3.12/site-packages/tcim_lite --houmo /usr/local/houmo-sdk
  $0 -b build -c Release -j 8 --tcim $DADAO_VENV/lib/python3.12/site-packages/tcim_lite --houmo /usr/local/houmo-sdk --install /opt/hmcheck
  $0 -b build -c Release -j 8 --tcim $DADAO_VENV/lib/python3.12/site-packages/tcim_lite --houmo /usr/local/houmo-sdk --no-install

Notes:
  By default the script will read [TCIM_RUNTIME_PATH](file:///data/weiguo.xing/repo/imodelzoo/hmatc/setup.py#L29-L29) and `HOUMO_SDK_PATH` from environment.
  If they are not set, the script will attempt to auto-detect common installation paths.
  You can also explicitly provide paths via `--tcim` and `--houmo`.
  Installation runs by default unless --no-install is specified.
EOF
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -s) SOURCE_DIR="$2"; shift 2;;
    -b) BUILD_DIR="$2"; shift 2;;
    -c) CONFIG="$2"; shift 2;;
    -g) GENERATOR="$2"; shift 2;;
    -j) JOBS="$2"; shift 2;;
    --tcim) TCIM="$2"; shift 2;;
    --houmo) HOUMO="$2"; shift 2;;
    --install) 
      if [[ -n "${2:-}" ]] && [[ ! "$2" =~ ^- ]]; then
        INSTALL_DIR="$2"
        shift 2
      else
        INSTALL_DIR="${SOURCE_DIR}/../bin"
        shift 1
      fi
      INSTALL=1
      ;;
    --no-install) 
      INSTALL=0
      shift 1
      ;;
    -h|--help) usage; exit 0;;
    --) shift; break;;
    *) echo "Unknown arg: $1"; usage; exit 1;;
  esac
done

command -v cmake >/dev/null 2>&1 || { echo "cmake not found in PATH" >&2; exit 1; }

if [[ -n "$TCIM" ]]; then
  export TCIM_RUNTIME_PATH="$TCIM"
fi
if [[ -n "$HOUMO" ]]; then
  export HOUMO_SDK_PATH="$HOUMO"
fi

# If environment vars are not set, try to auto-detect common installation paths
if [[ -z "${TCIM_RUNTIME_PATH:-}" ]]; then
  candidates=("$DADAO_VENV/lib/python3.12/site-packages/tcim_lite")
  for p in "${candidates[@]}"; do
    if [[ -d "$p" && ( -d "$p/include" || -d "$p/inc" ) ]]; then
      export TCIM_RUNTIME_PATH="$p"
      echo "Auto-detected TCIM_RUNTIME_PATH: $p"
      break
    fi
  done
fi
if [[ -z "${TCIM_RUNTIME_PATH:-}" ]]; then
  echo "Environment variable TCIM_RUNTIME_PATH is not set and auto-detection failed.
Please set TCIM_RUNTIME_PATH (e.g. export TCIM_RUNTIME_PATH=$DADAO_VENV/lib/python3.12/site-packages/tcim_lite) or pass --tcim /path." >&2
  exit 1
fi

if [[ -z "${HOUMO_SDK_PATH:-}" ]]; then
  candidates=("/usr/local/houmo-sdk")
  for p in "${candidates[@]}"; do
    if [[ -d "$p" && ( -d "$p/include" || -d "$p/inc" ) ]]; then
      export HOUMO_SDK_PATH="$p"
      echo "Auto-detected HOUMO_SDK_PATH: $p"
      break
    fi
  done
fi
if [[ -z "${HOUMO_SDK_PATH:-}" ]]; then
  echo "Environment variable HOUMO_SDK_PATH is not set and auto-detection failed.
Please set HOUMO_SDK_PATH (e.g. export HOUMO_SDK_PATH=/usr/local/houmo-sdk) or pass --houmo /path." >&2
  exit 1
fi

echo "Source: $SOURCE_DIR"
echo "Build dir: $BUILD_DIR"
echo "Configuration: $CONFIG"
echo "Generator: ${GENERATOR:-(default)}"
echo "Jobs: $JOBS"
echo "TCIM_RUNTIME_PATH: $TCIM_RUNTIME_PATH"
echo "HOUMO_SDK_PATH: $HOUMO_SDK_PATH"

mkdir -p "$BUILD_DIR"

cmake_args=( -S "$SOURCE_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$CONFIG" )
if [[ -n "$GENERATOR" ]]; then
  cmake_args+=( -G "$GENERATOR" )
fi

echo "Running: cmake ${cmake_args[*]}"
cmake "${cmake_args[@]}"

echo "Building..."
cmake --build "$BUILD_DIR" --config "$CONFIG" -- -j "$JOBS"

if [[ "$INSTALL" -ne 0 ]]; then
  if [[ -z "$INSTALL_DIR" ]]; then
    INSTALL_DIR="${SOURCE_DIR}/../bin"
  fi
  
  # Determine if we need elevated privileges based on target directory
  NEED_SUDO=0
  if [[ "$INSTALL_DIR" == /usr/* ]] || [[ "$INSTALL_DIR" == /opt/* ]]; then
    NEED_SUDO=1
  fi
  
  # Create the install directory
  if [[ $NEED_SUDO -eq 1 ]]; then
    sudo mkdir -p "$INSTALL_DIR"
  else
    mkdir -p "$INSTALL_DIR"
  fi
  
  echo "Installing to: $INSTALL_DIR"
  
  # Use CMAKE_INSTALL_PREFIX to specify exact installation directory
  if [[ $NEED_SUDO -eq 1 ]]; then
    sudo cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR" --config "$CONFIG"
  else
    cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR" --config "$CONFIG"
  fi
fi

echo "Build complete. Executable should be in: $BUILD_DIR/"