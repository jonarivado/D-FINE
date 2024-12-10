"""
Copied from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import PIL
import PIL.Image

from typing import Any, Dict, List, Optional

from .._misc import convert_to_tv_tensor, _boxes_keys
from .._misc import Image, Video, Mask, BoundingBoxes, BoundingBoxFormat
from .._misc import SanitizeBoundingBoxes

from ...core import register
torchvision.disable_beta_transforms_warning()

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from torchvision.ops import box_convert
from PIL import Image as PILImage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
plt.switch_backend('TkAgg')

RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
RandomZoomOut = register()(T.RandomZoomOut)
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
Resize = register()(T.Resize)
# ToImageTensor = register()(T.ToImageTensor)
# ConvertDtype = register()(T.ConvertDtype)
# PILToTensor = register()(T.PILToTensor)
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
RandomCrop = register()(T.RandomCrop)
Normalize = register()(T.Normalize)


@register()
class EmptyTransform(T.Transform):
    def __init__(self, ) -> None:
        super().__init__()

    def forward(self, *inputs):
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


@register()
class PadToSize(T.Pad):
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )
    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        sp = F.get_spatial_size(flat_inputs[0])
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)  # type: ignore[arg-type]

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (
        BoundingBoxes,
    )
    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt
        self.normalize = normalize

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt


@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (
        PIL.Image.Image,
    )
    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        inpt = F.pil_to_tensor(inpt)
        if self.dtype == 'float32':
            inpt = inpt.float()

        if self.scale:
            inpt = inpt / 255.

        inpt = Image(inpt)

        return inpt

@register()
class AlbumentationsTransform(T.Transform):
    def __init__(self):
        super().__init__()
        self.transform = A.Compose([
            A.Downscale(scale_min=0.125, scale_max=0.999, p=0.75),
            A.RandomRotate90(p=0.5),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Blur(p=0.1),
            A.MedianBlur(p=0.2),
            A.ToGray(p=0.01),
            A.CLAHE(p=0.01),
            A.ImageCompression(quality_lower=75, p=0.1),
        ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))

    def forward(self, inpt):
        # Unpack the input tuple
        image, target, metadata = inpt

        # Convert PIL Image to NumPy array
        image_np = np.array(image)

        # Extract bounding boxes and labels
        bboxes = target['boxes'].numpy()  # Access the tensor from BoundingBoxes
        labels = target['labels'].numpy()

        # Convert tensors to lists for Albumentations
        bboxes = bboxes.tolist()
        labels = labels.tolist()

        # Apply Albumentations transformations
        transformed = self.transform(image=image_np, bboxes=bboxes, labels=labels)

        # Convert transformed image back to PIL Image
        transformed_image = PILImage.fromarray(transformed['image'])
        

        # Convert bounding boxes and labels back to tensors
        transformed_bboxes = torch.tensor(transformed['bboxes'], dtype=torch.float32)
        # transformed_bboxes = F.to_tensor(transformed_bboxes)
        transformed_labels = torch.tensor(transformed['labels'], dtype=torch.int64)
        # transformed_labels = F.to_tensor(transformed_bboxes)

        # Reconstruct the BoundingBoxes object
        canvas_size = transformed_image.size[::-1]  # (height, width)
        transformed_bboxes = BoundingBoxes(
            transformed_bboxes,
            format=BoundingBoxFormat.XYXY,
            canvas_size=canvas_size
        )
        transformed_image = F.to_tensor(transformed_image)

        # Update target with transformed data
        target['boxes'] = transformed_bboxes
        target['labels'] = transformed_labels

        # self.plot_and_save(transformed_image, transformed_bboxes, metadata)

        return transformed_image, target, metadata
    
    
    def plot_and_save(self, image, bboxes, metadata, save_path='transformed_image.png'):
            # Convert tensor image to numpy array
            image_np = image.permute(1, 2, 0).numpy()

            # Create a figure and axis
            fig, ax = plt.subplots(1)

            # Display the image
            ax.imshow(image_np)

            # Add bounding boxes
            for bbox in bboxes:
                x_min, y_min, x_max, y_max = bbox
                width = x_max - x_min
                height = y_max - y_min
                rect = patches.Rectangle((x_min, y_min), width, height, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)

            # Save the figure
            plt.savefig(save_path)
            plt.close(fig)