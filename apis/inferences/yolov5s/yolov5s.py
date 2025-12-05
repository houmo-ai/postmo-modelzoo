import os
import sys
import numpy as np
import logging
import time
import argparse

import cv2
import torch
import torchvision
import tcim_lite as tcim

HOUMO_TARGET = os.getenv("HOUMO_TARGET")
assert HOUMO_TARGET in ["xh2"], f"Unsupported HOUMO_TARGET: {HOUMO_TARGET}"

coco80_labels = [
    "person",
    "bicycle",
    "car",
    "motorbike",
    "aeroplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "sofa",
    "pottedplant",
    "bed",
    "diningtable",
    "toilet",
    "tvmonitor",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--enable_ort', action="store_true", help="use onnxruntime to post process."
    )
    args = parser.parse_args()
    return args


def infer_detect_onnx(onnx_path, inputs_list):
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path)
    input_names = list()
    input_dict = dict()
    for idx, input in enumerate(session.get_inputs()):
        input_name = input.name
        input_dict[input_name] = inputs_list[idx]
        input_names.append(input_name)

    outputs = session.run(None, input_dict)
    print("post-processing model output num:", len(outputs))
    tensor_res = torch.tensor(outputs[0])

    return tensor_res


def letterbox(
    im,
    new_shape=(640, 640),
    color=(114, 114, 114),
    auto=True,
    scaleFill=False,
    scaleup=True,
    stride=32,
):
    """
    Code from
    https://github.com/ultralytics/yolov3/blob/92c3bd7a4e997e215c7b3ec8bd5a3f9337d39776/utils/augmentations.py#L91

    """
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    # only scale down, do not scale up (for better val mAP)
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(
        im,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=color,
    )  # add border
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y


def box_area(box):
    # box = xyxy(4,n)
    return (box[2] - box[0]) * (box[3] - box[1])


def box_iou(box1, box2, eps=1e-7):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
        eps
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    (a1, a2), (b1, b2) = box1[:, None].chunk(2, 2), box2.chunk(2, 1)
    inter = (torch.min(a2, b2) - torch.max(a1, b1)).clamp(0).prod(2)

    # IoU = inter / (area1 + area2 - inter)
    return inter / (box_area(box1.T)[:, None] + box_area(box2.T) - inter + eps)


class YoloV5:
    def __init__(self, image_size=(640, 640), conf_threshold=0.25, iou_threshold=0.45):
        self._image_size = image_size
        self._conf_threshold = conf_threshold
        self._iou_threshold = iou_threshold

    def preprocess(self, image):
        out, _, _ = letterbox(image, self._image_size, stride=64, auto=False)  # HWC
        out = np.transpose(out, (2, 0, 1))  # CHW .astype(np.float32)
        out = np.expand_dims(out, axis=0)  # NCHW
        return out

    def yolo_detect(self, feats):
        # in.shape = out.shape: 1x3x80x80x85 1x3x40x40x85 1x3x20x20x85
        output = []

        for i, feat in enumerate(feats):
            data = torch.tensor(feat)
            assert len(data.shape) == 5

            bs, channel, ny, nx, no = data.shape
            grid, anchor_grid = self._make_grid(nx, ny, i)

            data[..., 0:2] = (data[..., 0:2] * 2 - 0.5 + grid) * self.stride[i]  # xy
            data[..., 2:4] = (data[..., 2:4] * 2) ** 2 * anchor_grid  # wh

            output.append(data.reshape(bs, -1, no))

        return torch.concat(output, dim=1)

    def _make_grid(self, nx=20, ny=20, i=0):
        anchors = torch.tensor(
            [
                [10, 13, 16, 30, 33, 23],
                [30, 61, 62, 45, 59, 119],
                [116, 90, 156, 198, 373, 326],
            ]
        )

        self.stride = torch.tensor([8, 16, 32]).view(-1, 1, 1)
        anchors = anchors.view(3, 3, 2)
        anchors = anchors / self.stride

        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)])
        grid = torch.stack((xv, yv), 2).expand((1, 3, ny, nx, 2)).float()
        anchor_grid = (
            (anchors[i] * self.stride[i])
            .view((1, 3, 1, 1, 2))
            .expand((1, 3, ny, nx, 2))
            .float()
        )
        return grid, anchor_grid

    def non_max_suppression(
        self,
        prediction,
        classes=None,
        agnostic=False,
        multi_label=False,
        labels=(),
        max_det=300,
        nm=0,  # number of masks
    ):
        """Non-Maximum Suppression (NMS) on inference results to reject overlapping bounding boxes
        Returns:
            list of detections, on (n,6) tensor per image [xyxy, conf, cls]
        """
        conf_thres = self._conf_threshold
        iou_thres = self._iou_threshold
        bs = prediction.shape[0]  # batch size
        nc = prediction.shape[2] - nm - 5  # number of classes
        xc = prediction[..., 4] > conf_thres  # candidates

        # Checks
        assert (
            0 <= conf_thres <= 1
        ), f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
        assert (
            0 <= iou_thres <= 1
        ), f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

        # Settings
        # min_wh = 2  # (pixels) minimum box width and height
        max_wh = 7680  # (pixels) maximum box width and height
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 0.5 + 0.05 * bs  # seconds to quit after
        redundant = True  # require redundant detections
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
        merge = False  # use merge-NMS

        t = time.time()
        mi = 5 + nc  # mask start index
        output = [torch.zeros((0, 6 + nm), device=prediction.device)] * bs
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # confidence
            # Cat apriori labels if autolabelling
            if labels and len(labels[xi]):
                lb = labels[xi]
                v = torch.zeros((len(lb), nc + 5), device=x.device)
                v[:, :4] = lb[:, 1:5]  # box
                v[:, 4] = 1.0  # conf
                v[range(len(lb)), lb[:, 0].long() + 5] = 1.0  # cls
                x = torch.cat((x, v), 0)

            # If none remain process next image
            if not x.shape[0]:
                continue

            # Compute conf
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            box = xywh2xyxy(x[:, :4])
            mask = x[:, mi:]  # zero columns if no masks

            # Detections matrix nx6 (xyxy, conf, cls)
            if multi_label:
                i, j = (x[:, 5:mi] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, j + 5, None], j[:, None].float(), mask[i]), dim=1
                )
            else:  # best class only
                conf, j = x[:, 5:mi].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float(), mask), dim=1)[
                    conf.view(-1) > conf_thres
                ]

            # Filter by class
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            n = x.shape[0]  # number of boxes
            if not n:  # no boxes
                continue

            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
            i = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            i = i[:max_det]
            if merge and (1 < n < 3e3):  # Merge NMS (boxes merged using weighted mean)
                # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(
                    1, keepdim=True
                )  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy

            output[xi] = x[i]
            if (time.time() - t) > time_limit:
                logging.warning(f'WARNING: NMS time limit {time_limit:.3f}s exceeded')
                break  # time limit exceeded

        return output

    def scale_coords(self, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        if ratio_pad is None:  # calculate from img0_shape
            gain = min(
                self._image_size[0] / img0_shape[0], self._image_size[1] / img0_shape[1]
            )  # gain  = old / new
            pad = (self._image_size[1] - img0_shape[1] * gain) / 2, (
                self._image_size[0] - img0_shape[0] * gain
            ) / 2  # wh padding
        else:
            gain = ratio_pad[0][0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [1, 3]] -= pad[1]  # y padding
        coords[:, :4] /= gain
        # clip_coords(coords, img0_shape)
        return coords


if __name__ == '__main__':
    args = get_args()
    sys.path.insert(0, "../../common/python")
    print("\n===> yolov5s python example start...")
    print(
        f"tcim runtime version: {tcim.runtime.get_version()}, houmo target: {HOUMO_TARGET}, enable ort: {args.enable_ort}"
    )

    # 1. load model
    model_path = "./yolov5s_clip_xh2_b1_1core.hmm"
    module = tcim.runtime.load(model_path)

    # 2. preprocess
    yolov5 = YoloV5()
    img_path = "../../data/000000000139.jpg"
    cv_image = cv2.imread(img_path)

    input_data, _, _ = letterbox(cv_image, (640, 640), stride=64, auto=False)
    input_data = cv2.cvtColor(input_data, cv2.COLOR_BGR2RGB)
    mean_arr = np.array([0.0, 0.0, 0.0])
    std_arr = np.array([255.0, 255.0, 255.0])
    input_data = (input_data - mean_arr) / std_arr
    input_data = np.transpose(input_data, (2, 0, 1))  # CHW float32
    input_data = np.expand_dims(input_data, axis=0)
    input_data = input_data.astype(np.float16)

    # 3. set input
    input_num = module.get_num_inputs()
    for id in range(0, input_num):
        input_name = module.get_input_name(id)
        input_info = module.get_input_info(input_name).ascontiguous()
        print(
            "input[{}] shape = {}, dtype = {}, format = {}".format(
                input_name, input_info.shape, input_info.dtype, input_info.format.name
            )
        )
        module.set_input(input_name, input_data)

    # 4. run & sync
    module.run()
    module.sync()

    # 5. get output
    result_check = True
    outputs = []
    output_num = module.get_num_outputs()
    for id in range(0, output_num):
        output_name = module.get_output_name(id)
        output_info = (
            module.get_output_info(output_name).astype(np.float32).ascontiguous()
        )
        print(
            "output[{}] shape = {}, dtype = {}, format = {}".format(
                output_name,
                output_info.shape,
                output_info.dtype,
                output_info.format.name,
            )
        )
        output_data = module.get_output(output_name).astype(np.float32).numpy()
        outputs.append(output_data)

    # 6. postprocess
    assert len(outputs) == 3
    if args.enable_ort:
        # 6.1 use onnxruntime to post process
        onnx_path = "./yolov5s_640x640_postprocess.onnx"
        outputs = infer_detect_onnx(onnx_path, outputs)
    else:
        # 6.2 use torch func to post process
        outputs = yolov5.yolo_detect(outputs)

    outputs = yolov5.non_max_suppression(outputs)
    outputs = outputs[0]  # bs=1
    image_size = (cv_image.shape[0], cv_image.shape[1])
    outputs[:, :4] = yolov5.scale_coords(outputs[:, :4], image_size).round()
    boxes = outputs.numpy()

    # 7. print and draw
    print("box num = {}".format(len(boxes)))
    for det in boxes:
        (x1, y1, x2, y2), conf, cls = list(map(int, det[0:4])), det[4], int(det[5])
        # cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 1, 8)
        from ultralytics.utils.plotting import Annotator, colors

        label = coco80_labels[cls] + " {:.2f}".format(conf)
        annotator = Annotator(cv_image, line_width=2, example=str(cls))
        annotator.box_label((x1, y1, x2, y2), label, color=colors(cls, True))
        print(
            "x1:{}, y1:{}, x2:{}, y2:{}, conf:{:.6f}, cls:{}".format(
                x1, y1, x2, y2, conf, int(cls)
            ),
            flush=True,
        )
    save_results = "demo_results/python"
    if not os.path.exists(save_results):
        os.makedirs(save_results)
    filename = os.path.basename(img_path)
    save_path = os.path.join(save_results, filename)
    cv2.imwrite(save_path, cv_image)
    print("demo results saved to", save_path)
    # check result, modify it when you change model or data
    assert len(boxes) == 21

    print("<=== yolov5s python example completed.\n")
