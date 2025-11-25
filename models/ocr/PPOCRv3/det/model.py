import os
import cv2
import json
import time
import torch
import numpy as np
import importlib.util
if importlib.util.find_spec("shapely") is None:
    os.system("pip install shapely -i https://pypi.tuna.tsinghua.edu.cn/simple")
if importlib.util.find_spec("pyclipper") is None:
    os.system("pip install pyclipper -i https://pypi.tuna.tsinghua.edu.cn/simple")
from shapely.geometry import Polygon
import pyclipper

from tqdm import tqdm
from hmatc.utils import logger
from hmatc.base.base_model import BaseModel

class OCRDet(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.input_name = self.inputs_name[0]
        _, C, H, W = self.inputs_cfg[self.input_name]["shape"]
        self.input_size = (H, W)
        self.thresh = 0.3
        self.box_thresh = 0.6
        self.unclip_ratio = 1.5
        self.min_size = 3
        self.area_precision_constraint = 0.5
        self.iou_constraint = 0.5
        self.max_candidates = 100
        self.results = []
    
    def process_label(self, label_info):
        def expand_points_num(boxes):
            max_points_num = 0
            for box in boxes:
                if len(box) > max_points_num:
                    max_points_num = len(box)
            ex_boxes = []
            for box in boxes:
                ex_box = box + [box[-1]] * (max_points_num - len(box))
                ex_boxes.append(ex_box)
            return ex_boxes
        label = json.loads(label_info)
        nBox = len(label)
        boxes, txts, txt_tags = [], [], []
        for bno in range(0, nBox):
            box = label[bno]['points']
            txt = label[bno]['transcription']
            boxes.append(box)
            txts.append(txt)
            txt_tags.append(True if txt in ['*', '###'] else False)
        boxes = expand_points_num(boxes)
        boxes = np.array(boxes, dtype=np.float32)
        txt_tags = np.array(txt_tags, dtype=np.bool_)
        ret_data = {}
        ret_data['polys'] = boxes
        ret_data['texts'] = txts
        ret_data['ignore_tags'] = txt_tags
        return ret_data
    
    def get_mini_boxes(self, contour):
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [
            points[index_1], points[index_2], points[index_3], points[index_4]
        ]
        return box, min(bounding_box[1])
    
    def unclip(self, box, unclip_ratio):
        poly = Polygon(box)
        distance = poly.area * unclip_ratio / poly.length
        offset = pyclipper.PyclipperOffset()
        offset.AddPath(box, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(offset.Execute(distance))
        return expanded
    
    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype("int32"), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype("int32"), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype("int32"), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype("int32"), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype("int32"), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]
    
    def boxes_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        bitmap = _bitmap
        height, width = bitmap.shape

        outs = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST,
                                cv2.CHAIN_APPROX_SIMPLE)
        if len(outs) == 3:
            img, contours, _ = outs[0], outs[1], outs[2]
        elif len(outs) == 2:
            contours, _ = outs[0], outs[1]

        num_contours = min(len(contours), self.max_candidates)

        boxes = []
        scores = []
        for index in range(num_contours):
            contour = contours[index]
            points, sside = self.get_mini_boxes(contour)
            if sside < self.min_size:
                continue
            points = np.array(points)
            score = self.box_score_fast(pred, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue

            box = self.unclip(points, self.unclip_ratio).reshape(-1, 1, 2)
            box, sside = self.get_mini_boxes(box)
            if sside < self.min_size + 2:
                continue
            box = np.array(box)

            box[:, 0] = np.clip(
                np.round(box[:, 0] / width * dest_width), 0, dest_width)
            box[:, 1] = np.clip(
                np.round(box[:, 1] / height * dest_height), 0, dest_height)
            boxes.append(box.astype("int32"))
            scores.append(score)
        return np.array(boxes, dtype="int32"), scores
    
    def postprocess(self, outs, in_datas):
        pred = outs[next(iter(outs))]
        if isinstance(pred, torch.Tensor):
            pred = pred.numpy()
        pred = pred[:, 0, :, :]
        segmentation = pred > self.thresh

        boxes_batch = []
        for batch_index in range(pred.shape[0]):
            src_h, src_w, _ = in_datas[self.input_name].shape
            mask = segmentation[batch_index]
            boxes, scores = self.boxes_from_bitmap(pred[batch_index], mask, src_w, src_h)
            boxes_batch.append({'points': boxes})
        return boxes_batch
    
    def evaluate_image(self, gt, pred):
        def get_union(pD, pG):
            return Polygon(pD).union(Polygon(pG)).area

        def get_intersection_over_union(pD, pG):
            return get_intersection(pD, pG) / get_union(pD, pG)

        def get_intersection(pD, pG):
            return Polygon(pD).intersection(Polygon(pG)).area

        perSampleMetrics = {}
        matchedSum = 0
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        #recall = 0
        precision = 0
        detMatched = 0
        iouMat = np.empty([1,1])
        gtPols = []
        detPols = []
        gtPolPoints = []
        detPolPoints = []
        gtDontCarePolsNum = []
        detDontCarePolsNum = []
        pairs = []
        detMatchedNums = []
        evaluationLog = ""

        for n in range(len(gt)):
            points = gt[n]['points']
            dontCare = gt[n]['ignore']
            if not Polygon(points).is_valid:
                continue
            gtPol = points
            gtPols.append(gtPol)
            gtPolPoints.append(points)
            if dontCare:
                gtDontCarePolsNum.append(len(gtPols) - 1)
        evaluationLog += "GT polygons: " + str(len(gtPols)) + (
            " (" + str(len(gtDontCarePolsNum)) + " don't care)\n"
            if len(gtDontCarePolsNum) > 0 else "\n")
        for n in range(len(pred)):
            points = pred[n]['points']
            if not Polygon(points).is_valid:
                continue
            detPol = points
            detPols.append(detPol)
            detPolPoints.append(points)
            if len(gtDontCarePolsNum) > 0:
                for dontCarePol in gtDontCarePolsNum:
                    dontCarePol = gtPols[dontCarePol]
                    intersected_area = get_intersection(dontCarePol, detPol)
                    pdDimensions = Polygon(detPol).area
                    precision = 0 if pdDimensions == 0 else intersected_area / pdDimensions
                    if (precision > self.area_precision_constraint):
                        detDontCarePolsNum.append(len(detPols) - 1)
                        break  
        evaluationLog += "DET polygons: " + str(len(detPols)) + (
            " (" + str(len(detDontCarePolsNum)) + " don't care)\n"
            if len(detDontCarePolsNum) > 0 else "\n")          

        if len(gtPols) > 0 and len(detPols) > 0:
            # Calculate IoU and precision matrixs
            outputShape = [len(gtPols), len(detPols)]
            iouMat = np.empty(outputShape)
            gtRectMat = np.zeros(len(gtPols), np.int8)
            detRectMat = np.zeros(len(detPols), np.int8)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    pG = gtPols[gtNum]
                    pD = detPols[detNum]
                    iouMat[gtNum, detNum] = get_intersection_over_union(pD, pG)
            for gtNum in range(len(gtPols)):
                for detNum in range(len(detPols)):
                    if gtRectMat[gtNum] == 0 and detRectMat[
                            detNum] == 0 and gtNum not in gtDontCarePolsNum and detNum not in detDontCarePolsNum:
                        if iouMat[gtNum, detNum] > self.iou_constraint:
                            gtRectMat[gtNum] = 1
                            detRectMat[detNum] = 1
                            detMatched += 1
                            pairs.append({'gt': gtNum, 'det': detNum})
                            detMatchedNums.append(detNum)
                            evaluationLog += "Match GT #" + \
                                             str(gtNum) + " with Det #" + str(detNum) + "\n"
        numGtCare = (len(gtPols) - len(gtDontCarePolsNum))
        numDetCare = (len(detPols) - len(detDontCarePolsNum))
        if numGtCare == 0:
            # recall = float(1)
            precision = float(0) if numDetCare > 0 else float(1)
        else:
            # recall = float(detMatched) / numGtCare
            precision = 0 if numDetCare == 0 else float(detMatched) / numDetCare
        matchedSum += detMatched
        numGlobalCareGt += numGtCare
        numGlobalCareDet += numDetCare

        perSampleMetrics = {
            'gtCare': numGtCare,
            'detCare': numDetCare,
            'detMatched': detMatched,
        }
        return perSampleMetrics
    
    def det_metric(self, preds, batch):
        gt_polyons_batch = batch[1]
        ignore_tags_batch = batch[2]
        for pred, gt_polyons, ignore_tags in zip(preds, gt_polyons_batch, ignore_tags_batch):
            gt_info_list = [{'points': gt_polyon,
                             'text': '',
                             'ignore': ignore_tag} for gt_polyon, ignore_tag in zip(gt_polyons, ignore_tags)]
            det_info_list = [{'points': det_polyon,
                              'text': ''} for det_polyon in pred['points']]
            result = self.evaluate_image(gt_info_list, det_info_list)
            self.results.append(result)
    
    def get_metric(self):
        numGlobalCareGt = 0
        numGlobalCareDet = 0
        matchedSum = 0
        for result in self.results:
            numGlobalCareGt += result['gtCare']
            numGlobalCareDet += result['detCare']
            matchedSum += result['detMatched']

        methodRecall = 0 if numGlobalCareGt == 0 else float(
            matchedSum) / numGlobalCareGt
        methodPrecision = 0 if numGlobalCareDet == 0 else float(
            matchedSum) / numGlobalCareDet
        methodHmean = 0 if methodRecall + methodPrecision == 0 else 2 * \
                                                                    methodRecall * methodPrecision / (
                                                                            methodRecall + methodPrecision)
        methodMetrics = {
            'precision': methodPrecision,
            'recall': methodRecall,
            'hmean': methodHmean
        }
        self.results = []
        return methodMetrics
    
    def demo(self, filepaths: list):
        save_path = f"./vis_{self.backend}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        in_datas = dict()
        for idx, filepath in enumerate(filepaths):
            file_name = os.path.basename(filepath)
            cv_image = cv2.imread(filepath)
            if cv_image is None:
                logger.warning(f'{filepath} not exists or decode failed')
                continue
            in_datas[self.input_name] = cv_image
            logger.info(f'Image[{idx}] {filepath}')
            
            preds = self.run(in_datas)

            boxes = preds[0]["points"]
            for idx, box in enumerate(boxes):
                box = np.array(box).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(cv_image, [box], True, color=(255, 255, 0), thickness=2)
                logger.info(f"License plate location {idx}: [({box[0][0]}), ({box[1][0]}), ({box[2][0]}), ({box[3][0]})]")
            save_dir = os.path.join(save_path, file_name)
            cv2.imwrite(save_dir, cv_image)
            logger.info(f"Save results to {save_dir}.")

    def evaluate(self, dataset, num=0):
        img_path_list = dataset.get_datas(num)
        pbar = tqdm(total=len(img_path_list),
                    desc="eval:",
                    position=0,
                    leave=True)
        total_frame = 0
        for data_line in dataset.data_lines:
            data_line = data_line.decode('utf-8')
            substr = data_line.strip("\n").split("\t")
            file_name = substr[0]
            label = substr[1]
            img_path = os.path.join(dataset.img_dir, file_name)
            if img_path not in img_path_list:
                continue
            cv_image = cv2.imread(img_path)
            if cv_image is None:
                logger.warning(f'{img_path} not exists or decode failed')
                continue
            src_h, src_w, _ = cv_image.shape
            in_datas = dict()
            in_datas[self.input_name] = cv_image
            ratio_h = float(self.input_size[0]) / src_h
            ratio_w = float(self.input_size[1]) / src_w
            dst_data = self.process_label(label)
            dst_data['shape'] = np.array([src_h, src_w, ratio_h, ratio_w])
            data_list = [np.stack((dst_data[key_str],), axis=0) for key_str in ['shape', 'polys', 'ignore_tags']]
            # self.shape_list = data_list[0]
            post_result = self.run(in_datas)
            self.det_metric(post_result, data_list)
            pbar.update(1)
            total_frame += 1
        metric = self.get_metric()
        pbar.close()

        logger.info("metric eval ********************")
        for k, v in metric.items():
            logger.info('{}:{}'.format(k, v))
        return {
            "input_size": self.inputs_cfg[self.input_name]["shape"],
            "dataset": dataset.dataset_name,
            "num": total_frame,
            "hmean": f"{metric['hmean']:.6f}",
            "latency": f"{self.ave_latency_ms:.6f}",
        }

