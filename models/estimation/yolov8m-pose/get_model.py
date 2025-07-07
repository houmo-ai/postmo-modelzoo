import os
import argparse
import numpy as np
import onnx
import onnx_graphsurgeon as osg
from hmassist.utils.utils import get_file_from_jfrog

HOUMO_TARGET = os.getenv('HOUMO_TARGET', 'houmo')

def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--type',
        dest='model_type',
        type=str,
        default='raw',
        help='which model type to get, choise in [raw, quant, all]',
    )
    parser.add_argument(
        '--quant_model_dir',
        dest='quant_model_dir',
        type=str,
        default=os.path.join('output', HOUMO_TARGET, 'hmquant'),
        help='where to save quant_model',
    )
    parser.add_argument(
        '--model_dir',
        dest='model_dir',
        type=str,
        default='.',
        help='where to save downloaded model',
    )
    args = parser.parse_args()
    return args

def clip_model():
    model = onnx.load("yolov8m-pose.onnx")

    # 创建新的输出张量
    new_output1 = osg.Variable(
        name="output1",
        dtype=np.float32,
        shape=[1, 51, 8400]
    )
    new_output2 = osg.Variable(
        name="output2",
        dtype=np.float32,
        shape=[1, 1, 4, 8400]
    )
    new_output3 = osg.Variable(
        name="output3",
        dtype=np.float32,
        shape=[1, 1, 8400]
    )
    # 创建 Graph 对象
    graph = osg.import_onnx(model)
    for output in graph.outputs:
        if output.name == "output0":
            graph.outputs.remove(output)

    for node in graph.nodes:
        if node.name == "/model.22/Reshape_6":
            graph.nodes.remove(node)
    for node in graph.nodes:
        if node.name == "/model.22/dfl/Reshape_1":
            graph.nodes.remove(node)
    for node in graph.nodes:
        if node.name == "/model.22/Concat_7":
            graph.nodes.remove(node)

    for node in graph.nodes:
        if node.name == "/model.22/Concat":
            node.outputs.clear()
            node.outputs.append(new_output1)
            graph.outputs.append(new_output1)
        if node.name == "/model.22/dfl/conv/Conv":
            node.outputs.clear()
            node.outputs.append(new_output2)
            graph.outputs.append(new_output2)
        if node.name == "/model.22/Sigmoid":
            node.outputs.clear()
            node.outputs.append(new_output3)
            graph.outputs.append(new_output3)
    graph.cleanup()
    graph.toposort()

    # 保存修改后的模型
    new_model = osg.export_onnx(graph)
    onnx.save(new_model, "yolov8m-pose-clip.onnx")


if __name__ == '__main__':
    args = get_args()
    quant_model_dir = args.quant_model_dir
    model_type = args.model_type
    model_dir = args.model_dir
    raw_path = "http://10.10.1.53:8082/artifactory/toolchain/support/custom/saimo/yolov8m-pose.onnx"
    quant_path = "models/yolov8m/hmquant_yolov8m_20250315.zip"
    yolov8s_pose_data = "http://10.10.1.53:8082/artifactory/toolchain/support/custom/yolov8_pose_data.tar.gz"

    if model_type == "raw" or model_type == "all":
        file_path = get_file_from_jfrog(raw_path, model_dir)
        get_file_from_jfrog(yolov8s_pose_data, model_dir, extract_dir="./")
    # if model_type == "quant" or model_type == "all":
    #     get_file_from_jfrog(quant_path, model_dir, quant_model_dir)
    clip_model()
