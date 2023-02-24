# 工具链modelzoo

## 目录

|目录|说明|备注|
|---|---|---|
|Apollo|域控项目相关模型支持||

imodelzoo下的一级目录是按照项目区分，目前域控项目使用的都是Apollo的模型，如果example属于其他项目请另外创建一个一级目录

## 执行

每个目录下有一个run_all.sh脚本，可以执行对应目录下的所有示例
示例可能存在环境要求，具体请参加个目录下的README.md

## 如何添加example

以添加支持域控项目的example为例： 在Apollo下增加新的example目录名，example完成后请在Apollo/run_all.sh添加脚本，确保新添加的example可以被Apollo/run_all.sh执行。

为了保障example在开发环境，交付环境，ci环境中能正常执行，几个注意事项：

- 开发环境
   参见：http://gerrit.houmo.ai/plugins/gitiles/toolchain/itvm/+/refs/heads/main/hdpl/README.md
   环境变量较多

- 交付环境
   参见：http://gerrit.houmo.ai/plugins/gitiles/common/platform/+/refs/heads/master/toolchain/docs/userguide/tcim/source/pages/installGuide.rst
   环境中只有HOUMO_PATH这个环境变量

- ci环境
   环境变量在ci脚本中设置，目前${TVM_SO_PATH}目录下有libtvm.so等动态库。PYTHONPATH已经指向tvm的python module。
   如果需要更多支持，请咨询 杨鑫绵(xinmian.yang@houmo.ai)
