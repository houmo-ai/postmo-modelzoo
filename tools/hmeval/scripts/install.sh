#!/usr/bin/env bash
# Install script for hmeval (CI-friendly)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== hmeval Install Script ==="
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

echo "Installing dependencies..."
pip3 install -r requirements.txt

echo "Building hmeval wheel package..."
rm -rf dist build *.egg-info
python3 setup.py bdist_wheel

echo "Uninstalling existing hmeval..."
pip3 uninstall -y hmeval || true

echo "Installing hmeval from wheel..."
pip3 install dist/hmeval-*.whl

echo "=== Installation complete ==="