#!/usr/bin/env bash
set -e

WORK_PATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "${WORK_PATH}" || exit 1

# check transformers version
version=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
if [ $? -ne 0 ]; then
  echo "未找到transformers库, 正在安装..."
  python3 -m pip install transformers==4.51.0
else
    # compare version
    comp_result=$(printf '%s\n' "4.51.0" "$version" | sort -V | head -n1)
    if [[ "$comp_result" == "$version" && "$comp_result" != "4.51.0" ]]; then
      echo "transformers库版本 $version 低于4.51.0, 正在升级..."
      python3 -m pip install transformers==4.51.0
    else
      echo "transformers库版本 $version 已满足要求, 无需升级"
    fi
fi
# get final transformers version
final_version=$(python3 -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "当前transformers库版本: ${final_version:-未安装}"

arch=$(uname -m)
if [ "$arch" = "aarch64" ]; then
  export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libgomp.so.1
  # get test model: 2 cores
  python3 get_model.py --ncore 2
else
  # get test model
  python3 get_model.py
fi

# python example
python3 demo.py
