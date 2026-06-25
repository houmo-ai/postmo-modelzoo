# Copyright 2025 HOUMO AI
#
# File: demo.py
# Description:
#   Sam2 Segmentation demo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

import os
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

from sam2_engine import SAM2Engine


# 全局变量存储点击的点
clicked_point = None
point_selected = False

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SAM2 demo with HMM/ONNX backend")
    parser.add_argument(
        "--backend",
        type=str,
        default="hmm",
        choices=["hmm", "xh2", "onnx"],
        help="Inference backend. xh2 is an alias of hmm.",
    )
    parser.add_argument(
        "--encoder",
        type=str,
        default=None,
        help="Path to encoder model. Defaults depend on backend.",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default=None,
        help="Path to decoder model. Defaults depend on backend.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(os.getenv("HOUMO_EXAMPLES_PATH", ""), "data/pic/beach.jpeg"),
        help="Path to input image",
    )
    parser.add_argument(
        "--mode",
        type=int,
        default=2,
        choices=[0, 1, 2, 3],
        help="0: COCO evaluation, 1: interactive click, 2: preset point, 3: automatic all-object segmentation",
    )
    parser.add_argument(
        "--point",
        type=int,
        nargs=2,
        default=[800, 800],
        help="Point coordinates (x y)",
    )
    parser.add_argument(
        "--grid-size",
        type=int,
        default=8,
        help="Grid size for automatic segmentation",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=os.path.join(os.getenv("HOUMO_EXAMPLES_PATH", ""), "data/datasets/coco2017"),
        help="dataset root directory for evaluation",
    )
    parser.add_argument(
        "--eval-num",
        type=int,
        default=0,
        help="Number of COCO val images to evaluate. 0 means all images.",
    )
    parser.add_argument(
        "--max-ann-per-image",
        type=int,
        default=0,
        help="Maximum COCO annotations per image to evaluate. 0 means all annotations.",
    )
    parser.add_argument(
        "--eval-output-dir",
        type=str,
        default=None,
        help="Directory for COCO evaluation prediction files",
    )
    return parser.parse_args()


def default_model_paths(backend):
    if backend == "onnx":
        return (
            os.path.join(CURRENT_DIR, "sam2.1s_encoder.onnx"),
            os.path.join(CURRENT_DIR, "sam2.1s_decoder.onnx"),
        )
    return (
        "output/xh2/sam2.1s_encoder_xh2_b1_1core_O2.hmm",
        "output/xh2/sam2.1s_decoder_xh2_b1_1core_O2.hmm",
    )


def mask_iou(mask_a, mask_b):
    """计算两个二值 mask 的 IoU。"""
    a = mask_a > 127
    b = mask_b > 127
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return intersection / union


def infer_with_prompt(model, image_features, point_coords, point_labels, new_h, new_w, orig_h, orig_w):
    """使用指定 prompt 执行一次 decoder 推理。"""
    decoder_outs = model.decode(image_features, point_coords, point_labels)
    return model.postprocess(decoder_outs, new_h, new_w, orig_h, orig_w)


def infer_with_features(model, image_features, point_x, point_y, scale, new_h, new_w, orig_h, orig_w):
    """基于已编码图像特征和单个点击点执行分割。"""
    scaled_x = point_x * scale
    scaled_y = point_y * scale
    offset = 10
    coords = np.array([[
        [scaled_x, scaled_y],
        [scaled_x - offset, scaled_y - offset],
        [scaled_x + offset, scaled_y + offset],
    ]], dtype=np.float32)
    labels = np.array([[1, 1, 1]], dtype=np.float32)
    mask, _ = infer_with_prompt(
        model, image_features, coords, labels, new_h, new_w, orig_h, orig_w
    )
    return mask


def infer_single_point(model, image, point_x, point_y):
    """单点分割演示推理。"""
    image_tensor, scale, new_h, new_w, orig_h, orig_w = model.preprocess(image)
    image_features = model.encode(image_tensor)
    return infer_with_features(
        model, image_features, point_x, point_y, scale, new_h, new_w, orig_h, orig_w
    )


def infer_auto_all(model, image, grid_size=8, min_area_ratio=0.002, iou_threshold=0.85):
    """通过网格采样自动分割图像中的所有候选对象。"""
    image_tensor, scale, new_h, new_w, h, w = model.preprocess(image)
    image_features = model.encode(image_tensor)

    min_area = h * w * min_area_ratio
    candidates = []
    xs = np.linspace(w / (grid_size + 1), w * grid_size / (grid_size + 1), grid_size)
    ys = np.linspace(h / (grid_size + 1), h * grid_size / (grid_size + 1), grid_size)

    for y in ys:
        for x in xs:
            mask = infer_with_features(
                model, image_features, x, y, scale, new_h, new_w, h, w
            )
            area = int((mask > 127).sum())
            if area < min_area:
                continue
            candidates.append({"mask": mask, "area": area, "point": (int(x), int(y))})

    candidates.sort(key=lambda item: item["area"], reverse=True)
    selected = []
    for candidate in candidates:
        duplicate = False
        for item in selected:
            if mask_iou(candidate["mask"], item["mask"]) > iou_threshold:
                duplicate = True
                break
        if not duplicate:
            selected.append(candidate)

    return selected


def mouse_callback(event, x, y, flags, param):
    """鼠标回调函数，用于捕获点击事件。"""
    global clicked_point, point_selected
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)
        point_selected = True
        print(f"[info] Selected point: ({x}, {y})")


def visualize_segmentation(image, mask, point=None, alpha=0.3, color=(30, 144, 255)):
    """可视化分割结果：叠加颜色和边缘线。"""
    result = image.copy()
    mask_colored = np.zeros_like(image)
    mask_colored[mask > 127] = color
    result = cv2.addWeighted(result, 1 - alpha, mask_colored, alpha, 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 255), thickness=3)

    if point is not None:
        cv2.circle(result, (int(point[0]), int(point[1])), 8, (0, 255, 0), -1)
        cv2.circle(result, (int(point[0]), int(point[1])), 10, (255, 255, 255), 2)

    return result


def visualize_auto_segmentation(image, masks, alpha=0.45):
    """可视化自动分割结果，每个对象使用不同颜色。"""
    result = image.copy()
    rng = np.random.default_rng(0)

    for idx, item in enumerate(masks):
        mask = item["mask"]
        color = rng.integers(64, 256, size=3, dtype=np.uint8)
        mask_area = mask > 127
        result[mask_area] = (
            result[mask_area].astype(np.float32) * (1 - alpha)
            + color.astype(np.float32) * alpha
        ).astype(np.uint8)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (0, 255, 255), thickness=2)
        if contours:
            x, y, _, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cv2.putText(
                result,
                str(idx + 1),
                (x, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 255),
                2,
            )

    return result



def interactive_click_segmentation(image, model):
    """交互式点击分割。"""
    global clicked_point, point_selected

    name = "Click to Segment"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(name, mouse_callback)

    print("\n" + "=" * 60)
    print("Interactive single-point segmentation mode")
    print("=" * 60)
    print("Click on the object to segment")
    print("Press 'q' or ESC to quit")
    print("Press 'r' to reset the selected point")
    print("=" * 60 + "\n")

    display_image = image.copy()

    while True:
        cv2.imshow(name, display_image)
        key = cv2.waitKey(1) & 0xFF

        if point_selected and clicked_point is not None:
            print(f"\n[info] Processing clicked point ({clicked_point[0]}, {clicked_point[1]})...")
            try:
                mask = infer_single_point(model, image, clicked_point[0], clicked_point[1])
                display_image = visualize_segmentation(
                    image, mask, point=clicked_point, alpha=0.4, color=(0, 255, 0)
                )
                print("[info] Segmentation completed")

                result_image = cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB)
                plt.figure(figsize=(12, 8))
                plt.imshow(result_image)
                plt.title(
                    f"Segmentation Result - Point: ({clicked_point[0]}, {clicked_point[1]})",
                    fontsize=16,
                )
                plt.axis("off")
                result_filename = f"demo_{model.backend}_single_point_result_{clicked_point[0]}_{clicked_point[1]}.png"
                plt.savefig(result_filename, bbox_inches="tight", dpi=150)
                print(f"[info] Result saved to: {result_filename}")
                plt.show()
            except Exception as e:
                print(f"[error] Inference failed: {e}")
                import traceback
                traceback.print_exc()

            point_selected = False

        if key == ord("q") or key == 27:
            print("\n[info] Exiting")
            break
        elif key == ord("r"):
            display_image = image.copy()
            clicked_point = None
            point_selected = False
            print("[info] Reset completed. You can select another point")

    cv2.destroyAllWindows()


def preset_point_segmentation(image, model, point_coords):
    """预设点分割模式。"""
    print("\n" + "=" * 60)
    print("Preset point segmentation mode")
    print("=" * 60)
    print(f"[info] Point coordinates: {point_coords}")

    mask = infer_single_point(model, image, point_coords[0], point_coords[1])
    result_image = visualize_segmentation(
        image, mask, point=point_coords, alpha=0.4, color=(30, 144, 255)
    )
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(result_rgb)
    plt.title(f"Single Point Segmentation - Point: {point_coords}", fontsize=16)
    plt.axis("off")
    result_filename = f"demo_{model.backend}_single_point_result.png"
    plt.savefig(result_filename, bbox_inches="tight", dpi=150)
    print(f"[info] Result saved to: {result_filename}")
    plt.show()
    print("[info] Demo completed")


def auto_all_segmentation(image, model, grid_size=8):
    """自动分割图像中的所有候选对象。"""
    print("\n" + "=" * 60)
    print("Automatic full-image segmentation mode")
    print("=" * 60)
    print(f"[info] Grid sampling size: {grid_size}x{grid_size}")

    masks = infer_auto_all(model, image, grid_size=grid_size)
    print(f"[info] Automatic segmentation completed. Candidate masks: {len(masks)}")

    result_image = visualize_auto_segmentation(image, masks)
    result_rgb = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(12, 8))
    plt.imshow(result_rgb)
    plt.title(f"Automatic Segmentation - {len(masks)} masks", fontsize=16)
    plt.axis("off")
    result_filename = f"demo_{model.backend}_auto_all_result.png"
    plt.savefig(result_filename, bbox_inches="tight", dpi=150)
    print(f"[info] Result saved to: {result_filename}")
    plt.show()
    print("[info] Demo completed")


if __name__ == "__main__":
    args = parse_args()

    default_encoder, default_decoder = default_model_paths(args.backend)
    encoder_path = args.encoder or default_encoder
    decoder_path = args.decoder or default_decoder

    cv_image = None
    if args.mode in (1, 2, 3):
        if not os.path.exists(args.image):
            print(f"[error] Image not found: {args.image}")
            exit(-1)

        cv_image = cv2.imread(args.image)
        if cv_image is None:
            print(f"[error] Failed to read image: {args.image}")
            exit(-1)

        h, w = cv_image.shape[:2]
        print(f"[info] Image size: {w}x{h}")

    model = SAM2Engine(backend=args.backend).load(encoder_path, decoder_path)

    if args.mode == 1:
        interactive_click_segmentation(cv_image, model)  # type: ignore[arg-type]
    elif args.mode == 2:
        preset_point_segmentation(cv_image, model, args.point)  # type: ignore[arg-type]
    elif args.mode == 3:
        auto_all_segmentation(cv_image, model, grid_size=args.grid_size)  # type: ignore[arg-type]
    elif args.mode == 0:
        results = model.eval(
            args.dataset_dir,
            num=args.eval_num,
            max_ann_per_image=args.max_ann_per_image,
            output_dir=args.eval_output_dir,
        )
        print(f"[result] {results}")
    else:
        print("[error] Invalid mode. Please set --mode to 0, 1, 2, or 3")
