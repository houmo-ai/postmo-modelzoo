import os
import cv2
import numpy as np
import tvm.tcim as tcim

module = tcim.load_so("resnet50")

img_path = "../../data/datasets/imagenet/ILSVRC2012_val_00000004.JPEG"
if not os.path.exists(img_path):
    print("The img path not exist -> {}".format(img_path))
    exit(-1)
print("process: {}".format(img_path))
img = cv2.imread(img_path)
if img is None:
    print("Failed to load image -> {}".format(img_path))
    exit(-1)
if img.shape[1] > img.shape[0]:
    h = 256
    w = round(img.shape[1] / img.shape[0] * 256)
else:
    h = round(img.shape[0] / img.shape[1] * 256)
    w = 256
img = cv2.resize(img, (w, h))  # HWC
h1 = round((h-224)/2)
h2 = h1 + 224
w1 = round((w-224)/2)
w2 = w1 + 224
img = img[h1:h2, w1:w2]
img = np.transpose(img, (2, 0, 1)).astype(np.float32)  # CHW
from hmassist.utils.transform import BGR2YUV
import torch
rgb2yuv_func = BGR2YUV(fmt="422")
img0 = rgb2yuv_func(torch.tensor(img)).numpy()  # HWC
# img = np.transpose(img, (2, 0, 1))  # CHW
img0 = np.expand_dims(img0, 0).astype(np.uint8)  # NCHW

from torchvision.datasets.folder import pil_loader
img1 = pil_loader(img_path)
import torchvision.transforms as transforms
from hmassist.utils.transform import RGB2YUV
from hmassist.utils.transform import ToTensorNotNormal
transform = transforms.Compose(
    [
        transforms.Resize(256), transforms.CenterCrop(224),
        ToTensorNotNormal(), RGB2YUV(),
    ],
)
img1 = transform(img1)
img1 = np.expand_dims(img1.numpy().astype(np.uint8), 0)

module.set_input("input.1", img0, "YUV422SP")
module.run()

outputs = {}
output_num = module.get_num_outputs()
for id in range(0, output_num):
    name = module.get_output_name_by_index(id)
    outputs[name] = module.get_float_output_by_name(name).numpy()

# postprocess
for name, data in outputs.items():
    from hmassist.utils.postprocess import softmax
    output = softmax(data)
    max_idx = np.argmax(output, axis=1).flatten()[0]
    max_prob = output.flatten()[max_idx]
    print("predict cls = {}, prob = {:.6f}".format(max_idx, max_prob))
