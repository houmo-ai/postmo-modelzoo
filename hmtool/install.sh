#!/bin/bash

set -e  # 出错就退出

PACKAGE_NAME="hmtool"

# 清理旧构建产物
rm -rf build dist 

# 构建 .whl 文件
python3 setup.py bdist_wheel

rm -rf "$PACKAGE_NAME.egg-info"

# 判断包是否已安装，再卸载
if python3 -c "import pkg_resources; pkg_resources.get_distribution('$PACKAGE_NAME')" 2>/dev/null; then
    echo "Uninstalling existing $PACKAGE_NAME ..."
    pip3 uninstall -y "$PACKAGE_NAME"
else
    echo "$PACKAGE_NAME is not installed, skipping uninstall."
fi

# 安装新版本
pip3 install dist/*.whl
