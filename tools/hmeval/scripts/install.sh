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

echo "Installing hmeval in develop mode..."
python setup.py develop

echo "=== Installation complete ==="