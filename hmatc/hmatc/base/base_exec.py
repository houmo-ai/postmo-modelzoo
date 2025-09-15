import abc
import os
import sys
import importlib
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from ..utils import logger
from ..utils.utils import get_onnx_inputs_info


class BaseExec(object, metaclass=abc.ABCMeta):
    def __init__(self, cfg: dict) -> None:
        """init parameters
        Args:
            cfg (dict): 来自配置文件
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        self.target = cfg["target"]
        self.model_cfg = cfg.get("model")
        self.model_path = self.model_cfg.get("model_path", "")
        HOUMO_MODEL_PATH = os.environ.get("HOUMO_MODEL_PATH", "")
        if not os.path.isfile(self.model_path):
            new_model_path = os.path.join(HOUMO_MODEL_PATH, self.model_path)
            if not os.path.isfile(new_model_path):
                logger.error(f"Not found model_path: {self.model_path}")
                exit(-1)
            self.model_path = new_model_path
        logger.info(f"model_path: {self.model_path}")
        self.model_name = self.model_cfg.get("name", "model")  # 编译后模型名称
        self.model_dir_name = Path.cwd().name  # 模型所在目录名称
        self.save_dir = self.model_cfg.get("save_dir")
        self.inputs_cfg = self.model_cfg.get("inputs")
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
        self.is_multi_input_model = len(self.inputs_cfg) > 1
        # 检查必须保证每个输入的batch相同
        self.model_input_batch = self.model_inputs_batch[
            self.inputs_name[0]
        ]  # 用户配置的输入batch
        # if self.is_multi_input_model:
        #     for idx in range(1, len(self.inputs_name)):
        #         batch = self.model_inputs_batch[self.inputs_name[idx]]
        #         if self.model_input_batch != batch:
        #             logger.error("all input batch must be same")
        #             exit(-1)
        self.quant_cfg = cfg.get("quant")
        self.calib_method = self.quant_cfg.get("calib_method", "minmax")
        if self.calib_method not in [
            "minmax",
            "kl",
            "percent-0.99",
            "mse",
            "ema",
            "aciq",
        ]:
            logger.error(f"calib_method {self.calib_method} is not supported")
            exit(-1)
        self.calib_data = self.quant_cfg["calib_data"]
        self.build_cfg = cfg.get("build")
        self.build_batch = self.build_cfg.get("batch", 1)
        self.build_ncore = self.build_cfg.get("ncore", 1)
        self.build_opt_level = self.build_cfg.get("opt_level", 2)
        self.build_opt_level = f"O{self.build_opt_level}"
        self.use_random_data = self.calib_data is None
        # 图像单输入，非图像or多输入必须是npz数据
        self.is_image_single_input = (
            not self.is_multi_input_model and self.data_formats[0] is not None
        )
        # resizer工作模式
        # 0 - 输入为非图像数据or多输入情况，禁用resizer，相当于非图像输入
        # 1 - 全动态resizer，参数为10个值[y, x, height, width, h, w, top, left, bottom, right]
        # 2 - crop部分动态resizer, 参数为4个值[y, x, height, width]
        # 3 - 静态resizer，使用场景几乎没有，不建议用
        self.resizer_mode = 0
        self.custom_msg = dict()
        for name in self.inputs_cfg:
            input_cfg = self.inputs_cfg[name]
            self.custom_msg[name] = dict(
                shape=input_cfg["shape"],
                resizer_mode=self.resizer_mode,
                input_cfg=input_cfg,
            )
        self.roi_num = 1
        self.onnx_inputs_info, self.onnx_outputs_info = get_onnx_inputs_info(
            self.model_path
        )
        self.onnx_is_static = True
        for input_name in self.inputs_name:
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
        self.outputs_name = list()
        for name in self.onnx_outputs_info:
            self.outputs_name.append(name)
        self.demo_cfg = cfg.get("demo", dict())
        self.eval_cfg = cfg.get("eval", dict())

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
        model_path, warmup_num, sample_num, loop_num=1, device_id=1, thread_num=1
    ):
        from ..python import perf

        # TODO 使用golden数据
        perf_info = perf.CModelRunner(
            model_path, sample_num, thread_num, device_id, loop_num, warmup_num
        )
        t_start = datetime.now().strftime("%Y%m%d%H%M%S")
        res_info = {
            "perf": {
                t_start: {
                    "params": {
                        "hmm_path": model_path,
                        "thread_num": thread_num,
                        "device_num": device_id,
                        "loop_num": loop_num,
                        "warmup_num": warmup_num,
                        "sample_num": sample_num,
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
