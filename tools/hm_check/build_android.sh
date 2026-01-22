#!/usr/bin/env bash
# build_android.sh - cross-compile hm-check for Android using Android NDK + CMake
# Usage:
#   ./build_android.sh -b build_android -c Release -j 8 --ndk /path/to/ndk --tcim /opt/tcim --houmo /opt/houmo --abi arm64-v8a --platform android-35 --install
set -euo pipefail

SOURCE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SOURCE_DIR}/build_android"
CONFIG=Release
GENERATOR="Ninja"
JOBS=$(nproc 2>/dev/null || echo 4)
NDK="${NDK_PATH:-}"
TCIM=""
HOUMO=""
ABI="arm64-v8a"
PLATFORM="android-35"
INSTALL=1  # Changed from 0 to 1 to enable default installation
INSTALL_DIR=""  # Initialize INSTALL_DIR variable

usage() {
  cat <<EOF
Usage: $0 [options]

Options:
  -b DIR         Build directory (default: ./build_android)
  -c CONFIG      Build configuration: Release or Debug (default: Release)
  -g GENERATOR   CMake generator (default: Ninja)
  -j JOBS        Parallel jobs for build (default: number of CPUs)
  --ndk PATH     Path to Android NDK (default: uses NDK_PATH env or common locations)
  --tcim PATH    Set TCIM_RUNTIME_PATH (default: read from env TCIM_RUNTIME_PATH)
  --houmo PATH   Set HOUMO_SDK_PATH (default: read from env HOUMO_SDK_PATH)
  --abi ABI      Android ABI (default: arm64-v8a)
  --platform API Android platform level (default: android-35)
  --install [PATH]  Install after building (default: ${SOURCE_DIR}/../android, always enabled by default)
  --no-install   Skip installation after building
  -h             Show this help

Example:
  ./build_android.sh -b build_android -c Release -j 8 --ndk /opt/android-ndk-r25 --tcim /opt/tcim --houmo /opt/houmo --abi arm64-v8a --platform android-35
  ./build_android.sh -b build_android -c Release -j 8 --ndk /opt/android-ndk-r25 --tcim /opt/tcim --houmo /opt/houmo --abi arm64-v8a --platform android-35 --install /opt/custom-android
  ./build_android.sh -b build_android -c Release -j 8 --ndk /opt/android-ndk-r25 --tcim /opt/tcim --houmo /opt/houmo --abi arm64-v8a --platform android-35 --no-install

Notes:
  - Script prefers environment variables [TCIM_RUNTIME_PATH](file:///data/weiguo.xing/repo/imodelzoo/hmatc/setup.py#L29-L29) and `HOUMO_SDK_PATH` if set.
  - If `--ndk` is not provided, script tries $NDK_PATH and common installation paths.
  - Installation runs by default unless --no-install is specified.
EOF
}

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    -b) BUILD_DIR="$2"; shift 2;;
    -c) CONFIG="$2"; shift 2;;
    -g) GENERATOR="$2"; shift 2;;
    -j) JOBS="$2"; shift 2;;
    --ndk) NDK="$2"; shift 2;;
    --tcim) TCIM="$2"; shift 2;;
    --houmo) HOUMO="$2"; shift 2;;
    --abi) ABI="$2"; shift 2;;
    --platform) PLATFORM="$2"; shift 2;;
    --install) 
      if [[ -n "${2:-}" ]] && [[ ! "$2" =~ ^- ]]; then
        INSTALL_DIR="$2"
        shift 2
      else
        INSTALL_DIR="${SOURCE_DIR}/../android"
        shift 1
      fi
      INSTALL=1
      ;;
    --no-install) 
      INSTALL=0
      shift 1
      ;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1"; usage; exit 1;;
  esac
done

command -v cmake >/dev/null 2>&1 || { echo "cmake not found in PATH" >&2; exit 1; }
command -v realpath >/dev/null 2>&1 || { echo "realpath not found in PATH" >&2; exit 1; }

# prefer env variables if provided
if [[ -n "${TCIM:-}" ]]; then
  export TCIM_RUNTIME_PATH="$TCIM"
fi
if [[ -n "${HOUMO:-}" ]]; then
  export HOUMO_SDK_PATH="$HOUMO"
fi

# find NDK
if [[ -z "${NDK:-}" ]]; then
  if [[ -n "${NDK_PATH:-}" ]]; then
    NDK="$NDK_PATH"
  else
    candidates=("$HOME/Android/Sdk/ndk-bundle" "$HOME/Android/Sdk/ndk" "/opt/android-ndk" "/usr/local/android-ndk")
    for p in "${candidates[@]}"; do
      if [[ -d "$p" ]]; then
        NDK="$p"
        break
      fi
    done
  fi
fi

if [[ -z "${NDK:-}" || ! -d "$NDK" ]]; then
  echo "Android NDK not found. Please set --ndk /path/to/ndk or set NDK_PATH environment variable." >&2
  exit 1
fi

# ensure TCIM/HOUMO
if [[ -z "${TCIM_RUNTIME_PATH:-}" ]]; then
  candidates=("/opt/tcim" "/usr/local/tcim" "${SOURCE_DIR}/../../tcim" )
  for p in "${candidates[@]}"; do
    if [[ -d "$p" && ( -d "$p/include" || -d "$p/inc" ) ]]; then
      export TCIM_RUNTIME_PATH="$p"
      echo "Auto-detected TCIM_RUNTIME_PATH: $p"
      break
    fi
  done
fi
if [[ -z "${TCIM_RUNTIME_PATH:-}" ]]; then
  echo "TCIM_RUNTIME_PATH not set and auto-detection failed. Set TCIM_RUNTIME_PATH or pass --tcim." >&2
  exit 1
fi

if [[ -z "${HOUMO_SDK_PATH:-}" ]]; then
  candidates=("/opt/houmo_sdk" "/usr/local/houmo_sdk" "${SOURCE_DIR}/../../houmo_sdk" )
  for p in "${candidates[@]}"; do
    if [[ -d "$p" && ( -d "$p/include" || -d "$p/inc" ) ]]; then
      export HOUMO_SDK_PATH="$p"
      echo "Auto-detected HOUMO_SDK_PATH: $p"
      break
    fi
  done
fi
if [[ -z "${HOUMO_SDK_PATH:-}" ]]; then
  echo "HOUMO_SDK_PATH not set and auto-detection failed. Set HOUMO_SDK_PATH or pass --houmo." >&2
  exit 1
fi

# print configuration
echo "NDK: $NDK"
echo "Source: $SOURCE_DIR"
echo "Build dir: $BUILD_DIR"
echo "Configuration: $CONFIG"
echo "Generator: $GENERATOR"
echo "Jobs: $JOBS"
echo "ANDROID_ABI: $ABI"
echo "ANDROID_PLATFORM: $PLATFORM"
echo "TCIM_RUNTIME_PATH: $TCIM_RUNTIME_PATH"
echo "HOUMO_SDK_PATH: $HOUMO_SDK_PATH"

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake_args=( -S "$SOURCE_DIR" -B "$BUILD_DIR" -G "$GENERATOR" \
    -DCMAKE_TOOLCHAIN_FILE="$NDK/build/cmake/android.toolchain.cmake" \
    -DANDROID_ABI="$ABI" -DANDROID_PLATFORM="$PLATFORM" -DANDROID_NDK="$NDK" \
    -DCMAKE_BUILD_TYPE="$CONFIG" )

# Add install prefix only if installation is enabled
if [[ "$INSTALL" -ne 0 ]]; then
  if [[ -z "$INSTALL_DIR" ]]; then
    INSTALL_DIR="${SOURCE_DIR}/../android"
  fi
  cmake_args+=( -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" )
else
  cmake_args+=( -DCMAKE_INSTALL_PREFIX="$SOURCE_DIR/android" )
fi

echo "Running: cmake ${cmake_args[*]}"
cmake "${cmake_args[@]}"

echo "Building..."
cmake --build "$BUILD_DIR" --config "$CONFIG" -- -j "$JOBS"

if [[ "$INSTALL" -ne 0 ]]; then
  if [[ -z "$INSTALL_DIR" ]]; then
    INSTALL_DIR="${SOURCE_DIR}/../android"
  fi
  
  echo "Installing to: $INSTALL_DIR"
  cmake --install "$BUILD_DIR" --prefix "$INSTALL_DIR" --config "$CONFIG"
  
  echo "Install completed to: $INSTALL_DIR"
fi

echo "Android build complete. Output (installed) under: ${INSTALL_DIR:-${SOURCE_DIR}/../android}"