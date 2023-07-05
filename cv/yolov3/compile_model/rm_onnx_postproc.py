import copy
import os

import onnx


def remove_yolo(dest_file_path, src_file_path):
    onnx_model = onnx.load(src_file_path)
    graph = onnx_model.graph
    nodes = graph.node
    graph.node.remove(nodes[207])
    for i in range(206, 196, -1):
        graph.node.remove(nodes[i])
    for i in range(193, 183, -1):
        graph.node.remove(nodes[i])
    for i in range(180, 170, -1):
        graph.node.remove(nodes[i])

    graph.output[0].name = '341'
    graph.output[0].type.tensor_type.elem_type = 1
    out_0 = graph.output[0]
    out_1 = copy.deepcopy(out_0)
    out_2 = copy.deepcopy(out_0)
    out_1.name = '325'
    out_2.name = '309'
    graph.output.append(out_1)
    graph.output.append(out_2)
    onnx.save(onnx_model, dest_file_path)


if __name__ == '__main__':
    onnx_model_path = os.path.join(
        os.environ.get('MODEL_PATH'), 'yolov3.onnx',
    )
    print(onnx_model_path)
    remove_yolo('quant_yolov3_without_yolo.onnx', onnx_model_path)
