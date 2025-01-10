import torch
from torchvision import transforms
import numpy as np
from typing import Any

# YUV = RGB * M[0:3] + M[3]
M_RGB2YUV = {
    # 'BT601': [
    #     [0.299, -0.168735892,  0.5],
    #     [0.587, -0.331264108, -0.418687589],
    #     [0.114,  0.5,         -0.081312411],
    #     [0, 128, 128] # bias
    # ],
    # Below is a lite version of above
    "BT601": [
        [0.299, -0.169, 0.5],
        [0.587, -0.331, -0.419],
        [0.114, 0.5, -0.081],
        [0, 128, 128],  # bias
    ],
}

# RBG = YUV * M[0:3] + M[3]
M_YUV2RGB = {
    # From Nvidia PVA
    "BT601": [
        [1, 1, 1],
        [0, -0.344, 1.772],
        [1.402, -0.714, 0],
        [-179.456, 135.459, -226.816],
    ],
}


M_BGR2YUV = {
    # Below is a lite version of above
    "BT601": [
        [0.114, 0.5, -0.081],
        [0.587, -0.331, -0.419],
        [0.299, -0.169, 0.5],
        [0, 128, 128],  # bias
    ],
}

# RBG = YUV * M[0:3] + M[3]
M_YUV2BGR = {
    # From Nvidia PVA
    "BT601": [
        [1, 1, 1],
        [1.772, -0.344, 0],
        [0, -0.714, 1.402],
        [-226.816, 135.459, -179.456],
    ],
}

class YUVFormat:
    def __init__(self, fmt="422", interpolation=False) -> None:
        self.fmt = fmt
        self._MAP = {
            "420": (2, 2),
            "422": (2, 1),
            "YUV420": (2, 2),
            "YUV422": (2, 1),
            "420SP": (2, 2),
            "422SP": (2, 1),
        }
        self.interpolation = interpolation

    def __call__(self, img: torch.Tensor):
        _, img_h, img_w = img.size()
        # breakpoint()
        y, u, v = torch.split(img, 1, dim=0)
        if self.fmt in ["444", "444SP"]:
            uv = torch.stack([u, v], dim=-1)
            y = y.view(-1)
            uv = uv.view(-1)
            yuv = torch.cat((y, uv), 0)
            return yuv.view((img_h, img_w, 3))
        div_w, div_h = self._MAP[self.fmt]
        if self.interpolation:
            uv_resize = transforms.Resize((img_h // div_h, img_w // div_w))
            u = uv_resize(u)

            v = uv_resize(v)
            # Convert u and v to uint8 with clipping and rounding:
            u = u.clip(0, 255).round()
            v = v.clip(0, 255).round()
        else:
            u = u[:, 0::div_w]
            u = u[0::div_h, :]
            v = v[:, 0::div_w]
            v = v[0::div_h, :]
        uv = torch.stack([u, v], dim=-1)
        y = y.view(-1)
        uv = uv.view(-1)
        yuv = torch.cat((y, uv), 0)
        result = torch.zeros(img_h * img_w * 3)
        result[: yuv.shape[0]] = yuv
        return result.view((img_h, img_w, 3))


class RGB2YUV:
    def __init__(self, version="BT601", fmt="422", interpolation=True) -> None:
        """
        layout hwc or chw
        """
        Mb = M_RGB2YUV[version]

        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3]).T
        self.b = self.b.view(3, 1, 1)
        self.formatter = YUVFormat(fmt, interpolation)

    def __call__(self, img: torch.Tensor) -> Any:
        # self.M.to(img.device)
        # self.b.to(img.device)
        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()

        # Change YUV store format
        result = self.formatter(result)
        return result

class BGR2YUV:
    def __init__(self, version="BT601", fmt="422", interpolation=True) -> None:
        """
        layout hwc or chw
        """
        Mb = M_BGR2YUV[version]

        self.M = torch.Tensor(Mb[0:3]).T
        self.b = torch.Tensor(Mb[3]).T
        self.b = self.b.view(3, 1, 1)
        self.formatter = YUVFormat(fmt, interpolation)

    def __call__(self, img: torch.Tensor) -> Any:
        # self.M.to(img.device)
        # self.b.to(img.device)
        result = torch.einsum("ij,jhw->ihw", [self.M, img])
        result = result + self.b
        result.clip_(0, 255).round_()

        # Change YUV store format
        result = self.formatter(result)
        return result

def _is_numpy(img: Any) -> bool:
    return isinstance(img, np.ndarray)


def _is_numpy_image(img: Any) -> bool:
    return img.ndim in {2, 3}

class ToTensorNotNormal:
    def __call__(self, pic):
        # TODO: The torchvision.transforms.functional_pil module is removed in 0.17**
        # if not(F_pil._is_pil_image(pic) or _is_numpy(pic)):
        #     raise TypeError('pic should be PIL Image or ndarray. Got {}'.format(type(pic)))

        if _is_numpy(pic) and not _is_numpy_image(pic):
            raise ValueError(
                "pic should be 2/3 dimensional. Got {} dimensions.".format(pic.ndim)
            )

        default_float_dtype = torch.get_default_dtype()

        if isinstance(pic, np.ndarray):
            # handle numpy array
            if pic.ndim == 2:
                pic = pic[:, :, None]

            img = torch.from_numpy(pic.transpose((2, 0, 1))).contiguous()
            # backward compatibility
            if isinstance(img, torch.ByteTensor):
                return img.to(dtype=default_float_dtype)
            else:
                return img

        try:
            import accimage
        except ImportError:
            accimage = None
        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros([pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic).to(dtype=default_float_dtype)

        # handle PIL Image
        mode_to_nptype = {"I": np.int32, "I;16": np.int16, "F": np.float32}
        img = torch.from_numpy(
            np.array(pic, mode_to_nptype.get(pic.mode, np.uint8), copy=True)
        )

        if pic.mode == "1":
            img = 255 * img
        img = img.view(pic.size[1], pic.size[0], len(pic.getbands()))
        # put it from HWC to CHW format
        img = img.permute((2, 0, 1)).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.to(dtype=default_float_dtype)
        else:
            return img

    def __repr__(self):
        return self.__class__.__name__ + "()"