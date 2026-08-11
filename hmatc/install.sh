#!/bin/bash

set -e

PACKAGE_NAME="hmatc"

SMI_ARG="--enable_smi_support"

if [ "$1" = "--disable_smi_support" ]; then
    SMI_ARG=""
fi

rm -rf build dist

python3 setup.py bdist_wheel $SMI_ARG

rm -rf "$PACKAGE_NAME.egg-info"

if python3 -c "import pkg_resources; pkg_resources.get_distribution('$PACKAGE_NAME')" 2>/dev/null; then
    echo "Uninstalling existing $PACKAGE_NAME ..."
    pip3 uninstall -y "$PACKAGE_NAME"
else
    echo "$PACKAGE_NAME is not installed, skipping uninstall."
fi

pip3 install dist/*.whl
