import cv2
import numpy as np
from tqdm import tqdm
from typing import Dict, Any
from hmtool.utils import logger
from hmtool.utils.postprocess import softmax
from hmtool.base.base_model import BaseModel
from hmtool.datasets.imagenet import ILSVRC2012_LABELS


class ViT(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]

    def postprocess(self, outs: Dict[str, np.ndarray], in_datas: Dict[str, np.ndarray]) -> Any:
        output_name = list(outs.keys())[0]
        out = softmax(outs[output_name], axis=1, keepdims=True)
        max_idxes = np.argmax(out, axis=1, keepdims=True)
        batch = max_idxes.shape[0]
        res = list()
        for i in range(batch):
            max_idx = max_idxes[i][0]
            max_score = out[i][max_idx]
            res.append((max_idx, max_score))  # (cls_idx, score)
        return res
        
    def demo(self, filepaths: list):        
        in_datas = dict()
        for idx, filepath in enumerate(filepaths):
            cv_image = cv2.imread(filepath)
            if cv_image is None:
                logger.warning(f'{filepath} not exists or decode failed')
                continue
            in_datas[self.input_name] = cv_image
            logger.info(f'[{idx}] {filepath}')
            outs = self.run(in_datas)
            # 只需取batch0
            out = outs[0]
            cls_idx = str(out[0])
            score = out[1]
            cls_name = ILSVRC2012_LABELS[cls_idx][0]
            logger.info(f'score: {score:.3f}, cls_idx: {cls_idx}, cls_name: {cls_name}')
    
    def evaluate(self, dataset, num=0):
        img_paths, labels = dataset.get_datas(num)
        in_datas = dict()
        top1_acc = 0
        for idx, img_path in enumerate(tqdm(img_paths)):
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f'{img_path} not exists or decode failed')
                continue
            in_datas[self.input_name] = cv_image
            logger.debug(f'[{idx}] {img_path}')
            outs = self.run(in_datas)
            out = outs[0]
            cls_idx = str(out[0])
            gt_idx = str(labels[idx])
            if cls_idx == gt_idx:
                top1_acc += 1
        return {
            'input_size': self.inputs_cfg[self.input_name]['shape'],
            'dataset': dataset.dataset_name,
            'num': len(img_paths),
            'top1_acc': f'{top1_acc / len(img_paths):.6f}',
            'latency_ms': f'{self.ave_latency_ms:.6f}',
        }    