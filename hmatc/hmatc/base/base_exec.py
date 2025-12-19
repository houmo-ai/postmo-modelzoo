import abc
import os
import sys
import onnx
import json
import importlib
import numpy as np
import torch
from onnx import StringStringEntryProto
from pathlib import Path
from datetime import datetime
from ..utils import logger
from ..utils.utils import get_onnx_inputs_info


QUANTIZATION_RANGES = {
    "int8": (-128, 127),  # 8位有符号整数
    "uint8": (0, 255),  # 8位无符号整数
    "int16": (-32768, 32767),  # 16位有符号整数
    "uint16": (0, 65535),  # 16位无符号整数
    "int32": (-2147483648, 2147483647),  # 32位有符号整数
}


class BaseExec(object, metaclass=abc.ABCMeta):
    def __init__(self, cfg: dict) -> None:
        """init parameters
        Args:
            cfg (dict): 来自配置文件
        """
        """"基础参数"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.target = cfg["target"]
        self.enable_upload = False
        """模型参数"""
        self.model_cfg = cfg.get("model")
        self.save_dir = self.model_cfg.get("save_dir")
        self.model_path = self.model_cfg.get("model_path", "")
        HOUMO_MODEL_PATH = os.environ.get("HOUMO_MODEL_PATH", "")
        if not os.path.isfile(self.model_path):
            new_model_path = os.path.join(HOUMO_MODEL_PATH, self.model_path)
            if not os.path.isfile(new_model_path):
                logger.error(f"Not found model_path: {self.model_path}")
                exit(-1)
            self.model_path = new_model_path
        logger.info(f"model_path: {self.model_path}")
        self.onnx_inputs_info, self.onnx_outputs_info = get_onnx_inputs_info(
            self.model_path
        )
        self.onnx_is_static = True  # 初始为True
        self.model_name = self.model_cfg.get("name", "model")  # 编译后模型名称
        self.model_dir_name = Path.cwd().name  # 模型所在目录名称
        self.inputs_cfg = self.model_cfg.get("inputs")
        self.check_input_shape()
        self.model_inputs_batch = dict()
        self.inputs_name = list()
        self.data_formats = list()
        self.inputs_shape = list()
        self.resize_types = list()
        for input_name in self.inputs_cfg:
            model_batch = self.inputs_cfg[input_name]["shape"][0]
            self.model_inputs_batch[input_name] = model_batch
            self.inputs_name.append(input_name)
            self.data_formats.append(self.inputs_cfg[input_name].get("data_format"))
            self.inputs_shape.append(self.inputs_cfg[input_name]["shape"])
            self.resize_types.append(self.inputs_cfg[input_name].get("resize_type"))
        self.model_input_batch = self.model_inputs_batch[self.inputs_name[0]]

        self.is_multi_input_model = len(self.inputs_cfg) > 1  # 是否多输入
        self.is_image_single_input = (
            not self.is_multi_input_model and self.data_formats[0] is not None
        )  # 是否单输入且为图像

        """量化参数"""
        self.quant_cfg = cfg.get("quant", dict())
        self.quant_advance_cfg = self.quant_cfg.get("config", dict())
        self.calib_data = self.quant_cfg.get("calib_data")
        self.use_random_data = self.calib_data is None  # 使用随机数据量化
        self.quant_output_dir = os.path.join(self.save_dir, self.target, "hmquant")
        self.hmm_save_dir = os.path.join(self.save_dir, self.target)
        if not os.path.exists(self.hmm_save_dir):
            os.makedirs(self.hmm_save_dir)
        self.quant_onnx_model_path = os.path.join(
            self.quant_output_dir, f"hmquant_{self.model_name}_with_act.onnx"
        )
        # resizer相关参数
        # 0 - 输入为非图像数据or多输入情况，禁用resizer，相当于非图像输入
        # 1 - 全动态resizer，参数为10个值[y, x, height, width, h, w, top, left, bottom, right]
        # 2 - crop部分动态resizer, 参数为4个值[y, x, height, width]
        # 3 - 静态resizer
        self.resizer_mode = 0
        self.resizers_cfg = list()
        self.enable_static_resizers = list()
        self.max_inputs_size = dict()
        for input_name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[input_name]
            data_format = input_cfg.get("data_format")
            # 非图像或禁用resizer
            if data_format is None or "resizer" not in input_cfg:
                continue
            _, _, H, W = input_cfg["shape"]
            resizer_cfg = input_cfg.get("resizer", dict())
            max_input_size = resizer_cfg.get("max_input_size", [H, W])
            enable_static_resizer = resizer_cfg.get("enable_static_resizer", True)
            self.resizers_cfg.append(resizer_cfg)
            self.enable_static_resizers.append(enable_static_resizer)
            self.max_inputs_size[input_name] = max_input_size
        if self.is_image_single_input and len(self.resizers_cfg) == 1:
            if self.resize_types[0] == 0:
                self.resizer_mode = 2  # no padding
            elif self.resize_types[0] == 1:
                self.resizer_mode = 1  # padding
            # XH2没有dynamic_v1
            if self.resizer_mode == 2 and self.target == "xh2":
                self.resizer_mode = 1
            if self.enable_static_resizers[0]:
                self.resizer_mode = 3
        logger.info(f"resizer_mode: {self.resizer_mode}")

        """编译参数"""
        self.build_cfg = cfg.get("build", dict())
        self.build_batch = self.build_cfg.get("batch", 1)
        self.build_ncore = self.build_cfg.get("ncore", 1)
        self.build_opt_level = self.build_cfg.get("opt_level", 2)
        self.build_opt_level = f"O{self.build_opt_level}"
        self.build_output_dir = os.path.join(self.save_dir, self.target, "tcim")
        self.hmm_batch = self.build_batch * self.model_input_batch
        # roi模式
        # 0 - 1图n框
        # 1 - n图n框，每图1框，比如：1图1框、2图2框、...
        self.roi_num = self.build_cfg.get("roi_num", 1)
        if self.is_image_single_input and len(self.resizers_cfg) == 0:
            self.roi_num = 1
        if not isinstance(self.roi_num, int) or self.roi_num < 1:
            logger.error("roi_num must be int, and >= 1")
            exit(-1)
        if self.roi_num > 1 and self.hmm_batch > 1:
            logger.error(
                "Not support roi_num > 1, when model_input_batch > 1 or build_batch > 1"
            )
            exit(-1)
        if self.resizer_mode == 3:
            self.roi_num = 1
        if self.resizer_mode == 0:
            roi_tag = ""
            prefix = ""
        elif self.resizer_mode == 1:
            roi_tag = f"_{self.roi_num}roi"
            prefix = "_dynamic_v2" if self.target == "xh1" else "_dynamic"
        elif self.resizer_mode == 2:
            roi_tag = f"_{self.roi_num}roi"
            prefix = "_dynamic_v1"
        elif self.resizer_mode == 3:
            roi_tag = "_1roi"
            prefix = "_static"
        self.hmm_name = f"{self.model_name}_{self.target}_b{self.hmm_batch}{roi_tag}_{self.build_ncore}core_{self.build_opt_level}{prefix}"
        self.hmm_path = os.path.join(self.hmm_save_dir, f"{self.hmm_name}.hmm")

        """用于透传预处理信息进hmm"""
        self.custom_msg = dict()
        for name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[name]
            self.custom_msg[name] = dict(
                shape=input_cfg["shape"],
                resizer_mode=self.resizer_mode,
                input_cfg=input_cfg,
            )

        """模型评估参数"""
        self.demo_cfg = cfg.get("demo", dict())
        self.eval_cfg = cfg.get("eval", dict())

        # 图优化模块
        if "app_onnx_opt" in cfg["model"]:
            from ..optimizer.onnx_opt_engine import HMAppOnnxOptConvert

            self.ApplicationOnnxOpt = HMAppOnnxOptConvert(cfg)

    def check_input_shape(self):
        """
        检查配置shape和onnx是否一致
        """
        for input_name in self.inputs_cfg:
            onnx_shape = self.onnx_inputs_info[input_name]["shape"]
            cfg_shape = self.inputs_cfg[input_name]["shape"]
            for idx, val in enumerate(onnx_shape):
                if (
                    val < 0
                    or val is None
                    or isinstance(val, str)
                    or not isinstance(val, int)
                ):
                    self.onnx_is_static = False
                else:
                    # 检查配置的shape和onnx是否一致
                    if val != cfg_shape[idx]:
                        logger.error(
                            f"onnx shape {onnx_shape} is not equal to cfg shape {cfg_shape}"
                        )
                        exit(-1)

    @staticmethod
    def dtype_transform(dtype):
        if dtype == "float32":
            return "Float32Feature"
        elif dtype == "float16":
            return "Float16Feature"
        elif dtype == "float64":
            return "Float64Feature"
        elif dtype == "int8":
            return "Int8Feature"
        elif dtype == "uint8":
            return "Uint8Feature"
        elif dtype == "int16":
            return "Int16Feature"
        else:
            logger.error(f"Not support dtype: {dtype}")
            exit(-1)

    @staticmethod
    def gen_random_data(shape, dtype):
        if dtype == "float32":
            random_data = np.random.uniform(low=0, high=128, size=shape).astype(
                dtype=dtype
            )  # 数值范围[0, 128)
        elif dtype == "float16":
            random_data = np.random.uniform(low=0, high=128, size=shape).astype(
                dtype=dtype
            )  # 数值范围[0, 128)
        elif dtype == "int16":
            random_data = np.random.randint(low=0, high=128, size=shape, dtype=dtype)
        elif dtype == "int32":
            random_data = np.random.randint(low=0, high=128, size=shape, dtype=dtype)
        elif dtype == "int64":
            random_data = np.random.randint(low=0, high=128, size=shape, dtype=dtype)
        elif dtype == "uint8":
            random_data = np.random.randint(low=0, high=255, size=shape, dtype=dtype)
        elif dtype == "bool":
            random_data = np.random.randint(low=0, high=2, size=shape, dtype=dtype)
        else:
            logger.error(f"Not support dtype: {dtype}")
            exit(-1)
        return random_data

    @abc.abstractmethod
    def quantize(self):
        """模型量化"""
        pass

    @abc.abstractmethod
    def build(self):
        """模型编译"""
        pass

    @abc.abstractmethod
    def compare(self):
        """模型相似度比较"""
        pass

    @staticmethod
    def import_py_module_from_file(module_path: str, module_cls: str):
        module_name = os.path.splitext(os.path.basename(module_path))[1]
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        if spec is None:
            logger.error(f"module spec is None -> {module_path}")
            exit(-1)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        if not hasattr(module, module_cls):
            logger.error(f"module_cls not found -> {module_cls}")
            exit(-1)
        return getattr(module, module_cls)

    def get_model(self, backend):
        """获取模型实例"""
        model_impl_module = self.model_cfg.get("model_impl_module")
        model_impl_cls = self.model_cfg.get("model_impl_cls")
        if model_impl_module is None or model_impl_cls is None:
            logger.error("model_impl_module or model_impl_cls is None")
            return None
        model_impl_module_path = f"{model_impl_module}.py"
        if not os.path.exists(model_impl_module_path):
            logger.error(f"model_impl_module not exists -> {model_impl_module_path}")
            return None
        Model = self.import_py_module_from_file(model_impl_module_path, model_impl_cls)
        logger.info(f"from {model_impl_module} import {model_impl_cls} successfully")
        return Model(
            inputs_cfg=self.inputs_cfg,
            is_image_single_input=self.is_image_single_input,
            resizer_mode=self.resizer_mode,
            roi_num=self.roi_num,
            backend=backend,
        )

    def get_dataset(self, data_dir):
        """获取数据集"""
        dataset_module = self.eval_cfg.get("dataset_module")
        dataset_cls = self.eval_cfg.get("dataset_cls")
        if dataset_module is None or dataset_cls is None:
            logger.error("dataset_module or dataset_cls is None")
            return None
        module_path = f"{dataset_module}.py"
        if not os.path.exists(module_path):
            logger.error(f"dataset_module not exists -> {dataset_module}")
            return None
        Dataset = self.import_py_module_from_file(module_path, dataset_cls)
        logger.info(f"from {dataset_module} import {dataset_cls} successfully")
        return Dataset(root_path=data_dir)

    def demo(self, backend, device_id=0):
        """Demo入口"""
        if not self.demo_cfg:
            logger.error("demo config not found")
            return {}
        data_dir = self.demo_cfg.get("data_dir", "")
        HOUMO_DATASETS_PATH = os.environ.get(
            "HOUMO_DATASETS_PATH", "/usr/local/src/houmo-modelzoo/data/datasets"
        )
        HM_data_dir = os.path.join(HOUMO_DATASETS_PATH, data_dir)
        if not os.path.isdir(data_dir) and not os.path.isdir(HM_data_dir):
            logger.error("data_dir must be a exist directory")
            return {}
        if not os.path.isdir(data_dir):
            data_dir = HM_data_dir
        logger.info(f"[demo] data_dir: {data_dir}")
        test_num = self.demo_cfg.get("num", 0)
        if not isinstance(test_num, int):
            logger.error(f"test_num must be int -> {test_num}")
            return {}
        if test_num < 0:
            logger.error(f"test_num must >= 0 -> {test_num}")
            return {}
        model = self.get_model(backend)
        if model is None:
            logger.error("Failed to get model")
            return {}
        filenames = os.listdir(data_dir)
        data_num = len(filenames)
        if test_num > 0 and test_num < data_num:
            filenames = filenames[:test_num]
        filepaths = list()
        for filename in filenames:
            filepath = os.path.join(data_dir, filename)
            if not os.path.exists(filepath):
                logger.warning(f"filepath not exists -> {filepath}")
                continue
            filepaths.append(filepath)
        model_path = self.model_path if backend == "onnx" else self.hmm_path
        model.load(model_path, device_id)
        model.demo(filepaths)
        model.unload()

    def evaluate(self, backend, device_id=0):
        """评估入口"""
        if not self.eval_cfg:
            logger.error("eval config not found")
            return {}
        data_dir = self.eval_cfg.get("data_dir", "")
        HOUMO_DATASETS_PATH = os.environ.get(
            "HOUMO_DATASETS_PATH", "/usr/local/src/houmo-modelzoo/data/datasets"
        )
        HM_data_dir = os.path.join(HOUMO_DATASETS_PATH, data_dir)
        if not os.path.isdir(data_dir) and not os.path.isdir(HM_data_dir):
            logger.error("data_dir must be a exist directory")
            return {}
        if not os.path.isdir(data_dir):
            data_dir = HM_data_dir
        logger.info(f"[eval] data_dir: {data_dir}")
        num = self.eval_cfg.get("num", 0)
        if not isinstance(num, int):
            logger.error(f"eval test_num must be int -> {num}")
            return {}
        if num < 0:
            logger.error(f"eval test_num must >= 0 -> {num}")
            return {}
        # 获取dataset
        dataset = self.get_dataset(data_dir)
        if dataset is None:
            logger.error("get_dataset failed")
            return {}
        # 获取模型
        model = self.get_model(backend)
        if model is None:
            logger.error("Failed to get model")
            return {}
        model_path = self.model_path if backend == "onnx" else self.hmm_path
        model.load(model_path, device_id)
        res = model.evaluate(dataset, num)
        model.unload()
        logger.info(f"{res}")
        return res

    @staticmethod
    def model_perf(
        model_path,
        warmup_num,
        sample_num,
        loop_num=1,
        thread_num=1,
        stream_num=0,
        devices=[0],
    ):
        from ..python import perf

        # TODO 使用golden数据
        perf_info = perf.CModelRunner(
            model_path,
            warmup_num,
            sample_num,
            loop_num,
            thread_num,
            stream_num=0,
            check_output=False,
            devices=devices,
        )
        t_start = datetime.now().strftime("%Y%m%d%H%M%S")
        res_info = {
            "perf": {
                t_start: {
                    "params": {
                        "hmm_path": model_path,
                        "warmup_num": warmup_num,
                        "sample_num": sample_num,
                        "loop_num": loop_num,
                        "thread_num": thread_num,
                        "stream_num": stream_num,
                        "devices": devices,
                    },
                    "perf_info": {
                        "input_avg_latency": perf_info.input_avg_latency,
                        "input_max_latency": perf_info.input_max_latency,
                        "infer_avg_latency": perf_info.infer_avg_latency,
                        "infer_max_latency": perf_info.infer_max_latency,
                        "output_avg_latency": perf_info.output_avg_latency,
                        "output_max_latency": perf_info.output_max_latency,
                        "avg_cost": perf_info.avg_cost,
                        "qps": perf_info.qps,
                    },
                }
            }
        }
        return res_info

    def save_profile_data(self, outputs: dict):
        profile_dir = os.path.join(self.build_output_dir, "profile")
        if "auto_profile_data.bin" in outputs:
            os.makedirs(profile_dir, exist_ok=True)
            outputs["auto_profile_data.bin"].tofile(
                os.path.join(profile_dir, "auto_profile_data.bin")
            )
        if "primitive_profile_data.bin" in outputs:
            os.makedirs(profile_dir, exist_ok=True)
            outputs["primitive_profile_data.bin"].tofile(
                os.path.join(profile_dir, "primitive_profile_data.bin")
            )

    def add_node_output_as_graph_output(self, model_path):
        new_model_path = model_path.replace(".onnx", "_debug.onnx")
        # if os.path.exists(new_model_path):
        #     return new_model_path
        model = onnx.load(model_path)
        graph = model.graph
        for node in graph.node:
            if "constant" in str(node.op_type).lower():
                continue
            if self.target == "xh1":
                # 获取节点属性
                node_attributes = {}
                for attr in node.attribute:
                    value = onnx.helper.get_attribute_value(attr)
                    if isinstance(value, bytes):
                        value = value.decode("utf-8")
                    node_attributes[attr.name] = value
                if "output_granularity" not in node_attributes:
                    continue
                output_granularity = node_attributes["output_granularity"]
                output_dtype = node_attributes["output_dtype"]
                scale = node_attributes["output_scale"]
                zero_point = node_attributes["output_zero_point"]
            for output in node.output:
                if any(
                    output == existing_output.name for existing_output in graph.output
                ):
                    continue
                value_info = None
                for vi in graph.value_info:
                    if vi.name == output:
                        value_info = vi
                        break

                if value_info is None:
                    logger.warning(
                        f"Warning: Cannot determine type/shape for output {output}, creating basic output"
                    )
                    continue

                if self.target == "xh1":

                    def get_shape_from_value_info(value_info):
                        """从value_info中提取形状信息"""
                        if not value_info.type.HasField("tensor_type"):
                            return None

                        tensor_type = value_info.type.tensor_type
                        shape = []
                        if tensor_type.HasField("shape"):
                            for dim in tensor_type.shape.dim:
                                if dim.HasField("dim_value"):
                                    shape.append(dim.dim_value)
                                elif dim.HasField("dim_param"):
                                    shape.append(dim.dim_param)
                                else:
                                    shape.append("?")

                        return shape

                    output_shape = get_shape_from_value_info(value_info)
                    length = 1
                    for dim in output_shape:
                        length *= dim
                    if length == 1:
                        print(node.op_type, node.name, node_attributes)
                        continue
                    if output_granularity == "tensor":
                        block_shape = output_shape
                    else:
                        assert output_granularity.startswith("dim")
                        dim_index = int(output_granularity[-1])
                        block_shape = [
                            output_shape[i] if i != dim_index else 1
                            for i in range(len(output_shape))
                        ]
                    # add quantization info
                    quant_info = dict(
                        block_shape=block_shape,
                        scale=scale,
                        zero_point=zero_point,
                        src_dtype="float32",
                        dst_dtype=output_dtype,
                        dst_min=QUANTIZATION_RANGES[output_dtype][0],
                        dst_max=QUANTIZATION_RANGES[output_dtype][1],
                    )
                    quant_info_json = json.dumps(quant_info)
                    # 创建 StringStringEntryProto 对象
                    quant_entry = StringStringEntryProto(
                        key="houmo.quant.info", value=quant_info_json
                    )
                    # 将量化信息添加到 metadata_props
                    value_info.metadata_props.append(quant_entry)
                graph.output.append(value_info)

        model.graph.CopyFrom(graph)
        onnx.checker.check_model(model)
        model.opset_import[0].version = max(model.opset_import[0].version, 11)
        model.ir_version = max(model.ir_version, 6)
        onnx.save(model, new_model_path)
        logger.info(f"Saved debug model to {new_model_path}")
        return new_model_path
