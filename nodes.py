import cv2
import numpy as np
import torch
import torchvision

from .PixelOE.src.pixeloe.legacy.pixelize import pixelize
from .PixelOE.src.pixeloe.torch import env as pixeloe_env
from .PixelOE.src.pixeloe.torch.pixelize import pixelize as pixelize_torch
from .PixelOE.src.pixeloe.torch.utils import pre_resize


class PixelOE:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "mode": (["contrast", "center", "k-centroid", "bicubic", "nearest"],),
                "target_size": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "patch_size": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "pixel_size": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "thickness": ("INT", {
                    "default": 1,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "color_matching": ("BOOLEAN", {
                    "default": False,
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number",
                }),
                "colors": ("INT", {
                    "default": 256,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "number",
                }),
                "color_quant_method": (["kmeans", "maxcover"],),
                "colors_with_weight": ("BOOLEAN", {
                    "default": False,
                }),
                "no_upscale": ("BOOLEAN", {
                    "default": False,
                }),
                "no_downscale": ("BOOLEAN", {
                    "default": False,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"

    CATEGORY = "image/pixelize"

    @staticmethod
    def process(img, mode, target_size, patch_size, pixel_size, thickness,
                color_matching, contrast, saturation, colors, color_quant_method,
                colors_with_weight, no_upscale, no_downscale):

        if pixel_size == 0:
            pixel_size = None

        img = img.squeeze().numpy()
        img = (img * 255).astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        img_pix = pixelize(
            img=img, mode=mode, target_size=target_size, patch_size=patch_size, pixel_size=pixel_size, thickness=thickness,
            color_matching=color_matching, contrast=contrast, saturation=saturation, colors=colors, color_quant_method=color_quant_method,
            colors_with_weight=colors_with_weight, no_upscale=no_upscale, no_downscale=no_downscale,
        )

        img_pix = cv2.cvtColor(img_pix, cv2.COLOR_BGR2RGB)
        img_pix_t = np.array(img_pix).astype(np.float32) / 255.0
        img_pix_t = torch.from_numpy(img_pix_t)[None,]
        return (img_pix_t,)

class PixelOETorch:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "mode": (["contrast", "center", "k-centroid", "bicubic", "nearest"],),
                "target_size": ("INT", {
                    "default": 256,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "patch_size": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "pixel_size": ("INT", {
                    "default": 6,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "thickness": ("INT", {
                    "default": 3,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number",
                }),
                "do_color_match": ("BOOLEAN", {
                    "default": True,
                }),
                "do_quant": ("BOOLEAN", {
                    "default": False,
                }),
                "num_colors": ("INT", {
                    "default": 32,
                    "min": 1,
                    "max": 256,
                    "step": 1,
                    "display": "number",
                }),
                "quant_mode": (["kmeans", "weighted-kmeans", "repeat-kmeans"],),
                "dither_mode": (["ordered", "error_diffusion", "no"],),
                "torch_compile": ("BOOLEAN", {
                    "default": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process_torch"

    CATEGORY = "image/pixelize"

    @staticmethod
    def process_torch(img, target_size, patch_size, pixel_size, thickness, mode, do_color_match,
                    do_quant, num_colors, quant_mode, dither_mode, torch_compile):

        if pixel_size == 0:
            pixel_size = None

        pixeloe_env.TORCH_COMPILE = torch_compile

        img_t = img.squeeze().permute(2, 0, 1)
        img_pil = torchvision.transforms.functional.to_pil_image(img_t)
        img_t = pre_resize(
            img_pil=img_pil,
            target_size=target_size,
            patch_size=patch_size,
        )
        img_pix_t = pixelize_torch(
            img_t=img_t, pixel_size=pixel_size, thickness=thickness, mode=mode,
            do_color_match=do_color_match, do_quant=do_quant, num_colors=num_colors,
            quant_mode=quant_mode, dither_mode=dither_mode,
        )

        img_pix_t = img_pix_t.permute(0, 2, 3, 1)
        return (img_pix_t,)


NODE_CLASS_MAPPINGS = {
    "PixelOE": PixelOE,
    "PixelOETorch": PixelOETorch,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelOE": "PixelOE",
    "PixelOETorch": "PixelOETorch",
}
