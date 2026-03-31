#!/usr/bin/env bash
# Install script for hmeval (CI-friendly)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=== hmeval Install Script ==="
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Building hmeval wheel package..."
rm -rf dist build *.egg-info
python setup.py bdist_wheel

echo "Uninstalling existing hmeval..."
pip uninstall -y hmeval || true

echo "Installing hmeval from wheel..."
pip install dist/hmeval-*.whl

echo "=== Installation complete ==="