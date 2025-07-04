# 导出自定义多输入模型
python3 export_onnx.py
# 生成随机数据
python3 gen_data.py
# 模型量化
hmquant.sh
# 模型编译
hmbuild.sh