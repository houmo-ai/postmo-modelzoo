import onnx
import os
from ..utils import logger
import logging
import onnxruntime as ort

from .onnxOptimizer import OnnxOptimizer

work_level = {
    0: "product",
    1: "release",
    2: "debug",
    3: "ignore"
}

def restore_log():
    #### restore hmatc log level and handler
    from ..utils import console_handler
    for h in logger.handlers:
        logger.removeHandler(h)
    logger.addHandler(console_handler)
    logger.setLevel(logging.INFO)


class HMAppOnnxOptConvert(object):
    def __init__(self, cfg, custom_lib=""):
        self.opt_cfg = {}
        self.opt_status = cfg['model']['app_onnx_opt'].get('optimizer', False)
        self.opt_cfg['model_path'] = cfg['model']['model_path']
        self.opt_cfg['work_mode'] = work_level[int(cfg['model']['app_onnx_opt'].get('log_level', 0))]
        #self.opt_cfg['batch_size'] = cfg['batch']
        self.opt_cfg['out_path'] = cfg['model']['save_dir']
        if not os.path.exists(self.opt_cfg['out_path']):
            os.makedirs(self.opt_cfg['out_path'])
        self.opt_cfg['custom_lib'] = custom_lib
        self.opt_cfg['provider'] = ort.get_available_providers()
        self.opt_cfg['opt_version'] = "0.0.1"
        self.opt_cfg['base_opt'] = True
        self.opt_cfg['general_opt'] = True
        self.opt_cfg['model_name'] = self.get_model_name()
    
    def opt(self):
        if self.opt_status:
            src_model = onnx.load(self.opt_cfg['model_path'])
            onnx_opt = OnnxOptimizer(self.opt_cfg)
            dst_model, status = onnx_opt.run(src_model)
            if status:
                hmatc_metadata = {
                    "__set_flag__": "HMAppOpt",
                    "__producer__": "hmatc"
                }
                for key, value in hmatc_metadata.items():
                    entry = onnx.StringStringEntryProto()
                    entry.key = key
                    entry.value = value
                    dst_model.metadata_props.append(entry)
                onnx_out_path = os.path.join(self.opt_cfg["out_path"], f"{self.opt_cfg['model_name']}-opt.onnx")
                onnx.save(dst_model, onnx_out_path)
                logger.info(f"Houmo APP optimizer onnx model save to {onnx_out_path}")
                self.opt_model_path = onnx_out_path       

                restore_log()

    def get_model_name(self):
        return os.path.splitext(os.path.basename(self.opt_cfg['model_path']))[0]
        