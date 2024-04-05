import os

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from .PixelOE.pixeloe import pixelize

script_directory = os.path.dirname(os.path.abspath(__file__))


class PixelOE:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "img": ("IMAGE",),
                "mode": (["contrast"],),
                "target_size": ("INT", {
                    "default": 128,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "patch_size": ("INT", {
                    "default": 16,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "thickness": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "color_matching": ("BOOLEAN", {
                    "default": True
                }),
                "contrast": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "saturation": ("FLOAT", {
                    "default": 1.0,
                    "min": 0.0,
                    "max": 10.0,
                    "step": 0.01,
                    "round": 0.001,
                    "display": "number"
                }),
                "colors": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 4096,
                    "step": 1,
                    "display": "number"
                }),
                "no_upscale": ("BOOLEAN", {
                    "default": False
                }),
                "no_downscale": ("BOOLEAN", {
                    "default": False
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "process"

    CATEGORY = "Pixelize"

    def process(self, img, mode, target_size, patch_size, thickness, color_matching, contrast, saturation, colors, no_upscale, no_downscale):

        # Convert image from PyTorch tensor to NumPy array
        img = img.squeeze().numpy()

        # Convert image from float32 to uint8
        img = (img * 255).astype(np.uint8)

        # Convert image from RGB to BGR
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(img.shape)

        img_pix = pixelize(img, mode, target_size, patch_size, thickness, color_matching, contrast, saturation, colors, no_upscale, no_downscale)

        img_pix = cv2.cvtColor(img_pix, cv2.COLOR_BGR2RGB)
        img_pix_t = np.array(img_pix).astype(np.float32) / 255.0
        img_pix_t = torch.from_numpy(img_pix_t)[None,]
        return (img_pix_t,)


NODE_CLASS_MAPPINGS = {
    "PixelOE": PixelOE,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PixelOE": "PixelOE",
}