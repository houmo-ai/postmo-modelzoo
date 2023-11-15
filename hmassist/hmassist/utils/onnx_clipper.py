import os
import onnx

class OnnxNode:
    def __init__(self, node_map, onnx_model, onnx_node, node_name):
        if onnx_node is not None:
            self._onnx_node = onnx_node
        else:
            self._onnx_node = OnnxNode._find_node_by_name(
                onnx_model, node_name,
            )
        node_map[node_name] = self
        output_strs = self._onnx_node.output
        outputs = [
            node for node in onnx_model.graph.node if any(
                name in output_strs for name in node.input
            )
        ]
        self.output_names = [node.name for node in outputs]
        for _node in outputs:
            if _node.name not in node_map:
                OnnxNode(node_map, onnx_model, _node, _node.name)

    def get_output_names(self):
        return self.output_names

    def get_onnx_node(self):
        return self._onnx_node

    @classmethod
    def _find_node_by_name(cls, onnx_model, node_name):
        for node in onnx_model.graph.node:
            if node.name == node_name:
                return node
        print(f"Don't find {node_name} in onnx model")
        return None


def clip_onnx(onnx_model_path, config):
    if not os.path.isfile(onnx_model_path):
        print(f"[error] {onnx_model_path} is not a file.")
        exit()
    dir_name, file_fullname = os.path.split(onnx_model_path)
    file_name, file_ext = os.path.splitext(file_fullname)
    if not file_ext == '.onnx':
        print(f"[error] {onnx_model_path} is not an onnx file.")
        exit()
    onnx_model = onnx.load(onnx_model_path)
    # print(f"src_input = {onnx_model.graph.input}")
    # print(f"src_output = {onnx_model.graph.output}")
    onnx_node_map = {}
    to_be_removed_input_node = {}
    to_be_removed_node_name = set()
    for in_node in list(onnx_model.graph.input):
        first_nodes = [
            (node, node.name)
            for node in onnx_model.graph.node if in_node.name in node.input
        ]
        [
            OnnxNode(onnx_node_map, onnx_model, node, node.name)
            for node, node.name in first_nodes
        ]
        if in_node.name not in config['input']:
            to_be_removed_input_node[in_node.name] = in_node
            to_be_removed_node_name.update(
                [node.name for node, node.name in first_nodes],
            )
    for node_name in config['delete_node']:
        to_be_removed_node_name.add(node_name)

    final_remove_node_names = set()
    while len(to_be_removed_node_name) != 0:
        node_name = to_be_removed_node_name.pop()
        if node_name in final_remove_node_names:
            continue
        final_remove_node_names.add(node_name)
        my_onnx_node = onnx_node_map.get(node_name)
        if my_onnx_node is not None:
            to_be_removed_node_name.update(my_onnx_node.get_output_names())

    for _, node in to_be_removed_input_node.items():
        onnx_model.graph.input.remove(node)

    for node_name in final_remove_node_names:
        onnx_model.graph.node.remove(onnx_node_map[node_name].get_onnx_node())

    for node in list(onnx_model.graph.node):
        if node.op_type in config['delete_node_type']:
            onnx_model.graph.node.remove(node)

    for out_node in list(onnx_model.graph.output):
        if out_node.name not in config['output']:
            onnx_model.graph.output.remove(out_node)
    output_names = [node.name for node in onnx_model.graph.output]
    for output in config['output']:
        if output not in output_names:
            output_value_info = onnx.helper.make_tensor_value_info(
                output, onnx.TensorProto.FLOAT, shape=config['output'][output]['shape'],
            )
            onnx_model.graph.output.append(output_value_info)
    print(f"dest input = {onnx_model.graph.input}")
    print(f"dest output = {onnx_model.graph.output}")

    clip_name = file_name + '_clip'
    clip_path = os.path.join(dir_name, clip_name + '.onnx')
    onnx.save(onnx_model, clip_path)
    return onnx_model
