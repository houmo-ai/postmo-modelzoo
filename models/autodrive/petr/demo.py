import argparse
# from __future__ import division
import os
import time
import warnings
import sys

import torch
import numpy as np
import cv2
import pickle
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import tcim
from hmassist.utils.transform import BGR2YUV
from hmassist.utils.dist_metrics import cosine_distance
# from vis import vis_src_imgs, vis_bboxes, vis_src_imgs_and_bboxes
# sys.path.append(".")
# sys.path.append("./pytorch2onnx")

src_imgs = {
    0: '../../../data/datasets/nuscenes/samples/CAM_FRONT/n015-2018-07-11-11-54-16+0800__CAM_FRONT__1531281439762460.jpg',
    1: '../../../data/datasets/nuscenes/samples/CAM_FRONT_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_RIGHT__1531281439770339.jpg',
    2: '../../../data/datasets/nuscenes/samples/CAM_FRONT_LEFT/n015-2018-07-11-11-54-16+0800__CAM_FRONT_LEFT__1531281439754844.jpg',
    3: '../../../data/datasets/nuscenes/samples/CAM_BACK/n015-2018-07-11-11-54-16+0800__CAM_BACK__1531281439787525.jpg',
    4: '../../../data/datasets/nuscenes/samples/CAM_BACK_LEFT/n015-2018-07-11-11-54-16+0800__CAM_BACK_LEFT__1531281439797423.jpg',
    5: '../../../data/datasets/nuscenes/samples/CAM_BACK_RIGHT/n015-2018-07-11-11-54-16+0800__CAM_BACK_RIGHT__1531281439777893.jpg'
}


def get_args() -> argparse.Namespace:
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--batch',
        dest='batch',
        type=int,
        default=4,
        help='batch size',
    )
    args = parser.parse_args()
    return args


def vis_src_imgs():
    img_front_left = mpimg.imread(src_imgs[2])
    img_front = mpimg.imread(src_imgs[0])
    img_front_right = mpimg.imread(src_imgs[1])
    img_back_left = mpimg.imread(src_imgs[4])
    img_back = mpimg.imread(src_imgs[3])
    img_back_right = mpimg.imread(src_imgs[5])

    fig, axs = plt.subplots(2, 3, figsize=(20, 10))
    axs[0][0].imshow(img_front_left)
    axs[0][0].set_title('img_front_left')
    axs[0][0].axis('off')

    axs[0][1].imshow(img_front)
    axs[0][1].set_title('img_front')
    axs[0][1].axis('off')

    axs[0][2].imshow(img_front_right)
    axs[0][2].set_title('img_front_right')
    axs[0][2].axis('off')

    axs[1][0].imshow(img_back_left)
    axs[1][0].set_title('img_back_left')
    axs[1][0].axis('off')

    axs[1][1].imshow(img_back)
    axs[1][1].set_title('img_back')
    axs[1][1].axis('off')

    axs[1][2].imshow(img_back_right)
    axs[1][2].set_title('img_back_right')
    axs[1][2].axis('off')

    # 调整子图之间的间隔
    # left, right, top, bottom 参数分别控制子图边缘与图形窗口边缘的距离
    # wspace 和 hspace 参数控制子图之间的水平和垂直间距
    fig.subplots_adjust(wspace=0,hspace=0)  # 调整水平间距

    plt.show()
    if not os.path.exists('demo_results'):
        os.makedirs('demo_results')
    fig.savefig('demo_results/src_imgs.jpg', dpi=300)


def vis_bbox(axs, bbox,label,colors_vis):
    cx, cy, cz, w, l, h, rot, vx, vy = bbox
    # print(cx, cy, cz, w, l, h, rot, vx, vy)
    # print(cx+w/2, cy+h/2)
    # NOTE 右上角点、角度是逆时针为正
    rect = plt.Rectangle((cx+w/2, cy+h/2), w, l, angle = -np.degrees(rot), fill = False, color=colors_vis[label])
    axs.add_patch(rect)


def vis_bboxes(bboxes, labels, colors_vis, set_axis_equal=False, vis_x_range=(-50,50), vis_y_range=(-50,50)):
    # fig, axs = plt.subplots(figsize=(5, 10))
    fig, axs = plt.subplots()

    for bbox, label in zip(bboxes, labels):
      vis_bbox(axs, bbox, label, colors_vis)

    #Set the range and labels of the coordinate system
    axs.set_xlim(vis_x_range)
    axs.set_ylim(vis_y_range)
    axs.set_xlabel('X')
    axs.set_ylabel('Y')

    if set_axis_equal:
        plt.axis('equal')
    else:
        plt.xticks(np.arange(vis_x_range[0], vis_x_range[1], 10))
        plt.yticks(np.arange(vis_y_range[0], vis_y_range[1], 10))

    plt.show()
    fig.savefig('demo_results/bev_result.jpg', dpi=300)


def vis_src_imgs_and_bboxes(bboxes, labels, colors_vis, vis_x_range=(-50,50), vis_y_range=(-50,50)):
    img_front_left = mpimg.imread(src_imgs[2])
    img_front = mpimg.imread(src_imgs[0])
    img_front_right = mpimg.imread(src_imgs[1])
    img_back_left = mpimg.imread(src_imgs[4])
    img_back = mpimg.imread(src_imgs[3])
    img_back_right = mpimg.imread(src_imgs[5])

    fig, axs = plt.subplots(3, 3, figsize=(20, 10))
    axs[0][0].imshow(img_front_left)
    axs[0][0].set_title('img_front_left')
    axs[0][0].axis('off')

    axs[0][1].imshow(img_front)
    axs[0][1].set_title('img_front')
    axs[0][1].axis('off')

    axs[0][2].imshow(img_front_right)
    axs[0][2].set_title('img_front_right')
    axs[0][2].axis('off')

    axs[2][0].imshow(img_back_left)
    axs[2][0].set_title('img_back_left')
    axs[2][0].axis('off')

    axs[2][1].imshow(img_back)
    axs[2][1].set_title('img_back')
    axs[2][1].axis('off')

    axs[2][2].imshow(img_back_right)
    axs[2][2].set_title('img_back_right')
    axs[2][2].axis('off')

    # 调整子图之间的间隔
    # left, right, top, bottom 参数分别控制子图边缘与图形窗口边缘的距离
    # wspace 和 hspace 参数控制子图之间的水平和垂直间距
    fig.subplots_adjust(wspace=0, hspace=0.2, left=0.01, right=0.99, top=0.96, bottom=0.04)  # 调整水平间距

    for bbox, label in zip(bboxes, labels):
      vis_bbox(axs[1][1], bbox, label, colors_vis)

    #Set the range and labels of the coordinate system
    axs[1][1].set_xlim(vis_x_range)
    axs[1][1].set_ylim(vis_y_range)
    # axs[1][1].set_xlabel('X')
    # axs[1][1].set_ylabel('Y')
    axs[1][1].set_aspect('equal') 

    axs[1][0].axis('off')
    axs[1][2].axis('off')

    plt.show()
    if not os.path.exists('demo_results'):
        os.makedirs('demo_results')
    fig.savefig('demo_results/src_imgs_bev_result.jpg', dpi=300)


def vis_bev_image():
    img_front_left = cv2.imread(src_imgs[2])
    img_front = cv2.imread(src_imgs[0])
    img_front_right = cv2.imread(src_imgs[1])
    img_back_left = cv2.imread(src_imgs[4])
    img_back = cv2.imread(src_imgs[3])
    img_back_right = cv2.imread(src_imgs[5])

    bev_width = 900
    bev_height = 600
    bev_top_left = [0, bev_height]
    bev_top_right = [bev_width, bev_height]
    bev_bottom_right = [bev_width, 0]
    bev_bottom_left = [0, 0]
    bev_points = np.float32([bev_top_left, bev_top_right, bev_bottom_right, bev_bottom_left])

    # 定义透视变换的源点，这些点需要根据实际图像中车辆的位置来确定
    # 这里假设源图像和目标图像大小相同，源点为图像的四个角
    image_height, image_width = img_front.shape[:2]
    src_points = np.float32([[0, image_height], [image_width, image_height], [image_width, 0], [0, 0]])

    # 计算透视变换矩阵
    front_matrix = cv2.getPerspectiveTransform(src_points, bev_points)
    rear_matrix = cv2.getPerspectiveTransform(src_points, bev_points)
    left_matrix = cv2.getPerspectiveTransform(src_points, bev_points)
    right_matrix = cv2.getPerspectiveTransform(src_points, bev_points)
    top_matrix = cv2.getPerspectiveTransform(src_points, bev_points)
    bottom_matrix = cv2.getPerspectiveTransform(src_points, bev_points)

    # 对每个方向的图像进行透视变换
    bev_front = cv2.warpPerspective(img_front, front_matrix, (bev_width, bev_height))
    bev_back = cv2.warpPerspective(img_back, rear_matrix, (bev_width, bev_height))
    bev_front_left = cv2.warpPerspective(img_front_left, left_matrix, (bev_width, bev_height))
    bev_front_right = cv2.warpPerspective(img_front_right, right_matrix, (bev_width, bev_height))
    bev_back_left = cv2.warpPerspective(img_back_left, top_matrix, (bev_width, bev_height))
    bev_back_right = cv2.warpPerspective(img_back_right, bottom_matrix, (bev_width, bev_height))

    # 将所有鸟瞰图叠加在一起
    bev_combined = np.zeros((bev_height, bev_width, 3), dtype=np.uint8)
    cover_rate = 0.15
    cover_pixel = int(300 * cover_rate)

    scaled_front_left = cv2.resize(bev_front_left, (300, 300))
    bev_combined[:300, :300] = scaled_front_left

    scaled_front = cv2.resize(bev_front, (300, 300))
    bev_combined[:300, 300-cover_pixel:600-cover_pixel] = scaled_front

    scaled_front_right = cv2.resize(bev_front_right, (300, 300))
    bev_combined[:300, 600-cover_pixel:900-cover_pixel] = scaled_front_right

    scaled_back_left = cv2.resize(bev_back_left, (300, 300))
    scaled_back_left = cv2.flip(scaled_back_left, -1)
    bev_combined[300:600, :300] = scaled_back_left

    scaled_back = cv2.resize(bev_back, (300, 300))
    scaled_back = cv2.flip(scaled_back, -1)
    bev_combined[300:600, 300-cover_pixel:600-cover_pixel] = scaled_back

    scaled_back_right = cv2.resize(bev_back_right, (300, 300))
    scaled_back_right = cv2.flip(scaled_back_right, -1)
    bev_combined[300:600, 600-cover_pixel:900-cover_pixel] = scaled_back_right

    if not os.path.exists('demo_results'):
        os.makedirs('demo_results')
    cv2.imwrite('demo_results/bev_image.jpg', bev_combined)


def denormalize_bbox(normalized_bboxes):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]

    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


class NMSFreeCoder:
    """Bbox coder for NMS-free detector.
    Args:
        pc_range (list[float]): Range of point cloud.
        post_center_range (list[float]): Limit of the center.
            Default: None.
        max_num (int): Max number to be kept. Default: 100.
        score_threshold (float): Threshold to filter boxes based on score.
            Default: None.
        code_size (int): Code size of bboxes. Default: 9
    """

    def __init__(self,
                 pc_range,
                 voxel_size=None,
                 post_center_range=None,
                 max_num=100,
                 score_threshold=None,
                 num_classes=10):
        
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.post_center_range = post_center_range
        self.max_num = max_num
        self.score_threshold = score_threshold
        self.num_classes = num_classes

    def encode(self):
        pass

    def decode_single(self, cls_scores, bbox_preds):
        """Decode bboxes.
        Args:
            cls_scores (Tensor): Outputs from the classification head, \
                shape [num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            bbox_preds (Tensor): Outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        max_num = self.max_num

        cls_scores = cls_scores.sigmoid()
        scores, indexs = cls_scores.view(-1).topk(max_num)
        labels = indexs % self.num_classes
        bbox_index = indexs // self.num_classes
        bbox_preds = bbox_preds[bbox_index]

        final_box_preds = denormalize_bbox(bbox_preds)   
        final_scores = scores 
        final_preds = labels 

        # use score threshold
        if self.score_threshold is not None:
            thresh_mask = final_scores > self.score_threshold
        if self.post_center_range is not None:
            self.post_center_range = torch.tensor(self.post_center_range, device=scores.device)
            
            mask = (final_box_preds[..., :3] >=
                    self.post_center_range[:3]).all(1)
            mask &= (final_box_preds[..., :3] <=
                     self.post_center_range[3:]).all(1)

            if self.score_threshold:
                mask &= thresh_mask

            boxes3d = final_box_preds[mask]
            scores = final_scores[mask]
            labels = final_preds[mask]
            predictions_dict = {
                'bboxes': boxes3d,
                'scores': scores,
                'labels': labels
            }

        else:
            raise NotImplementedError(
                'Need to reorganize output as a batch, only '
                'support post_center_range is not None for now!')
        return predictions_dict

    def decode(self, preds_dicts):
        """Decode bboxes.
        Args:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, rot_sine, rot_cosine, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        Returns:
            list[dict]: Decoded boxes.
        """
        all_cls_scores = preds_dicts['all_cls_scores'][-1]
        all_bbox_preds = preds_dicts['all_bbox_preds'][-1]
        
        batch_size = all_cls_scores.size()[0]
        predictions_list = []
        for i in range(batch_size):
            predictions_list.append(self.decode_single(all_cls_scores[i], all_bbox_preds[i]))
        return predictions_list


def wrapped_post_process(cls_rsts,reg_rsts,reference,pc_range):
    # reference = inverse_sigmoid(reference_points.clone())
    outputs_classes = list()
    outputs_coords = list()
    for reg_rst,cls_rst in zip(reg_rsts,cls_rsts):
        reg_rst[..., 0:2] += reference[..., 0:2] # reference 加上cx,cy
        reg_rst[..., 0:2] = reg_rst[..., 0:2].sigmoid() # x,y进行sigmoid
        reg_rst[..., 4:5] += reference[..., 2:3] # reference 加上z
        reg_rst[..., 4:5] = reg_rst[..., 4:5].sigmoid() # z进行sigmoid
        outputs_coord = reg_rst
        outputs_classes.append(cls_rst)
        outputs_coords.append(outputs_coord)
    all_cls_scores = torch.stack(outputs_classes)
    all_bbox_preds = torch.stack(outputs_coords)
    # x,y,z都进行变换从而得到范围内的物体
    all_bbox_preds[..., 0:1] = (all_bbox_preds[..., 0:1] * (pc_range[3] - pc_range[0]) + pc_range[0])
    all_bbox_preds[..., 1:2] = (all_bbox_preds[..., 1:2] * (pc_range[4] - pc_range[1]) + pc_range[1])
    all_bbox_preds[..., 4:5] = (all_bbox_preds[..., 4:5] * (pc_range[5] - pc_range[2]) + pc_range[2])

    outs = {
        'all_cls_scores': all_cls_scores,
        'all_bbox_preds': all_bbox_preds,
        'enc_cls_scores': None,
        'enc_bbox_preds': None, 
    }
    return outs


def get_bboxes(preds_dicts, img_metas, max_num=300, score_threshold=0.6, num_classes=10, 
                voxel_size = [0.2, 0.2, 8], post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]):
    """Generate bboxes from bbox head predictions.
    Args:
        preds_dicts (tuple[list[dict]]): Prediction results.
        img_metas (list[dict]): Point cloud and image's meta info.
    Returns:
        list[dict]: Decoded bbox, scores and labels after nms.
    """
    bbox_coder = NMSFreeCoder(
    post_center_range=post_center_range,
    pc_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],
    max_num=max_num,
    voxel_size=voxel_size,
    score_threshold=score_threshold,
    num_classes=num_classes
    )
    preds_dicts = bbox_coder.decode(preds_dicts)
    num_samples = len(preds_dicts)

    ret_list = []
    for i in range(num_samples):
        preds = preds_dicts[i]
        bboxes = preds['bboxes']
        # cx, cy, cz, w, l, h, rot, vx, vy
        bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5
        # instance of class LiDARInstance3DBoxes
        # bboxes = img_metas[i]['box_type_3d'](bboxes, bboxes.size(-1))
        scores = preds['scores']
        labels = preds['labels']
        ret_list.append([bboxes, scores, labels])
    return ret_list


def infer(inputs, batch=1):
    part1 = tcim.runtime.load("petr_part1.hmm")
    for name in inputs:
        input_data = inputs[name]
        input_data = np.concatenate([input_data for i in range(batch)], axis=0)
        part1.set_input(name, input_data)

    part2 = tcim.runtime.load("petr_part2.hmm")
    part2_input_name = "pts_bbox_head_transformer_reshape_0_reshape"

    run_count = 50
    if os.getenv("HDPL_PLATFORM") == "ISIM":
        run_count = 1
    import time
    start_time = time.time()
    for i in range(run_count):
        part1.run()
        part1.sync()
        part1_output = part1.get_dev_output(part2_input_name)
        part2.set_input(part2_input_name, part1_output)
        part2.run()
        part2.sync()
    total_time = time.time() - start_time

    outputs = []
    for i in range(part2.get_num_outputs()):
        output_name = part2.get_output_name(i)
        output = part2.get_output(output_name, is_quanted=True)
        outputs.append(output)
        print("output[{}] name: {}, shape: {}".format(i, output_name, output.shape))

    print("=========run done=======")
    print("++++++++++++ total time is {} +++++++++++++++++".format(total_time))
    res_latency_per_batch = ((total_time) * 1000 / run_count) / batch
    throughput_per_six_batch = 1000 / res_latency_per_batch
    print('\033[92;20mInference average latency: %.3fms \033[0m' % res_latency_per_batch)
    print('\033[92;20mInference Throughput(QPS): %.2ffps \033[0m' % (throughput_per_six_batch))

    return outputs


def demo(batch=1):
    class_names = [
      'car', 'truck', 'trailer', 'bus', 'construction_vehicle', 'bicycle',
      'motorcycle', 'pedestrian', 'traffic_cone', 'barrier']
    colors_vis = ['b','g','r','c','m','y','k','#C5C5DC','#0AFAA4','#FF3BCD']

    data_list = []
    for i in src_imgs:
        data = cv2.imread(src_imgs[i])
        data = cv2.resize(data, (800, 450))  # HWC uint8
        data = data[130:450, :, :]
        data = np.transpose(data, (2, 0, 1))  # CHW uint8
        data = np.expand_dims(data, axis=0)  # NCHW uint8
        # convert to YUV
        data = torch.tensor(data.astype(np.float32))  # NHWC float32
        data = torch.squeeze(data, 0)  # HWC float32
        format = "YUV422"
        rgb2yuv_func = BGR2YUV(fmt=format)
        data = torch.unsqueeze(rgb2yuv_func(data), 0).numpy()  # NHWC float32
        data = data.astype(np.uint8)
        data_list.append(data)
    input_data = np.concatenate(data_list, axis=0)
    inputs = {"img": input_data}
    print("input_data shape={}".format(input_data.shape))

    outputs = infer(inputs, batch)

    reference = torch.from_numpy(np.load("reference.npy")) # 固定值
    out_0 = torch.from_numpy(outputs[0][:1])
    out_1 = torch.from_numpy(outputs[1][:1])

    print("output_0 shape={}".format(out_0.shape))
    print("output_1 shape={}".format(out_1.shape))

    # 反量化
    out_0 = out_0 * 0.0495
    out_1 = out_1 * 0.0214

    pc_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
    out_format = [[out_0], [out_1], reference, pc_range]
    out = wrapped_post_process(*out_format)

    bbox_list = get_bboxes(out, [], max_num=300, score_threshold=0.6, num_classes=len(class_names), 
                voxel_size = [0.2, 0.2, 8], post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0]) 

    # batch_size is 1
    bboxes, scores, labels = bbox_list[0]
    print(bboxes)
    print(scores)
    print(labels)
    # vis_src_imgs()
    # vis_bboxes(bboxes, labels, colors_vis, set_axis_equal=False)
    vis_src_imgs_and_bboxes(bboxes, labels, colors_vis)
    vis_bev_image()


if __name__ == '__main__':
    args = get_args()
    demo(args.batch)