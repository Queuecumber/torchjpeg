from __future__ import division

from typing import Any, List, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


class ImageList:
    """
    Structure that holds a list of images (of possibly
    varying sizes) as a single tensor.
    This works by padding the images to the same size,
    and storing in a field the original sizes of each image

    Attributes:
        image_sizes (list[tuple[int, int]]): each tuple is (h, w)

    Note:
        This class was taken from detectron2 (https://github.com/facebookresearch/detectron2/blob/master/detectron2/structures/image_list.py) with
        a small modification to the padding function which preserves gradients and some small fixes for linting and type checking.
        Otherwise the class and its documentation of unchanged.
    """

    def __init__(self, tensor: torch.Tensor, image_sizes: List[Tuple[int, int]]):
        """
        Arguments:
            tensor (Tensor): of shape (N, H, W) or (N, C_1, ..., C_K, H, W) where K >= 1
            image_sizes (list[tuple[int, int]]): Each tuple is (h, w). It can
                be smaller than (H, W) due to padding.
        """
        self.tensor = tensor
        self.image_sizes = image_sizes

    def __len__(self) -> int:
        return len(self.image_sizes)

    def __getitem__(self, idx) -> torch.Tensor:
        """
        Access the individual image in its original size.

        Args:
            idx: int or slice

        Returns:
            Tensor: an image of shape (H, W) or (C_1, ..., C_K, H, W) where K >= 1
        """
        size = self.image_sizes[idx]
        return self.tensor[idx, ..., : size[0], : size[1]]

    @torch.jit.unused
    def to(self, *args: Any, **kwargs: Any) -> "ImageList":
        """
        Implements the device API for ImageLists by copying the underyling storage to the target device

        Args:
            All arguments are forwarded to the underlying tensor storage :py:func:`torch.Tensor.to`

        Returns:
            ImageList: The imagelist object on the new device
        """
        cast_tensor = self.tensor.to(*args, **kwargs)
        return ImageList(cast_tensor, self.image_sizes)

    @property
    def device(self) -> torch.device:
        """
        Implements the device API for ImageLists by returning the underlying device

        Returns:
            torch.device: The device that the underlying tensor storage resides on
        """
        return self.tensor.device

    @staticmethod
    # https://github.com/pytorch/pytorch/issues/39308
    @torch.jit.unused
    def from_tensors(
        tensors: Sequence[torch.Tensor],
        size_divisibility: int = 0,
        pad_value: float = 0.0,
    ) -> "ImageList":
        """
        Args:
            tensors: a tuple or list of `torch.Tensors`, each of shape (Hi, Wi) or
                (C_1, ..., C_K, Hi, Wi) where K >= 1. The Tensors will be padded
                to the same shape with `pad_value`.
            size_divisibility (int): If `size_divisibility > 0`, add padding to ensure
                the common height and width is divisible by `size_divisibility`.
                This depends on the model and many models need a divisibility of 32.
            pad_value (float): value to pad

        Returns:
            an `ImageList`.
        """
        assert len(tensors) > 0
        assert isinstance(tensors, (tuple, list))
        for t in tensors:
            assert isinstance(t, torch.Tensor), type(t)
            assert t.shape[1:-2] == tensors[0].shape[1:-2], t.shape
        # per dimension maximum (H, W) or (C_1, ..., C_K, H, W) where K >= 1 among all tensors
        max_size = (
            # In tracing mode, x.shape[i] is Tensor, and should not be converted
            # to int: this will cause the traced graph to have hard-coded shapes.
            # Instead we should make max_size a Tensor that depends on these tensors.
            # Using torch.stack twice seems to be the best way to convert
            # list[list[ScalarTensor]] to a Tensor
            torch.stack([torch.stack([torch.as_tensor(dim) for dim in size]) for size in [tuple(img.shape) for img in tensors]])
            .max(0)
            .values
        )

        if size_divisibility > 1:
            stride = size_divisibility
            # the last two dims are H,W, both subject to divisibility requirement
            max_size = torch.cat([max_size[:-2], (max_size[-2:] + (stride - 1)) // stride * stride])

        image_sizes = [(int(im.shape[-2]), int(im.shape[-1])) for im in tensors]

        if len(tensors) == 1:
            # This seems slightly (2%) faster.
            image_size = image_sizes[0]
            padding_size = [
                0,
                int(max_size[-1] - image_size[1]),
                0,
                int(max_size[-2] - image_size[0]),
            ]
            if all(x == 0 for x in padding_size):  # https://github.com/pytorch/pytorch/issues/31734
                batched_imgs = tensors[0].unsqueeze(0)
            else:
                padded = F.pad(tensors[0], padding_size, value=pad_value)
                batched_imgs = padded.unsqueeze_(0)
        else:
            # max_size can be a tensor in tracing mode, therefore use tuple()
            batched_imgs = []
            for i, img in enumerate(tensors):
                image_size = image_sizes[i]
                padding_size = [
                    0,
                    int(max_size[-1] - image_size[1]),
                    0,
                    int(max_size[-2] - image_size[0]),
                ]

                padded = F.pad(img, padding_size, value=pad_value)
                batched_imgs.append(padded)

            batched_imgs = torch.stack(batched_imgs)

        return ImageList(batched_imgs.contiguous(), image_sizes)


def crop_batch(batch: Tensor, sizes: Tensor) -> Sequence[Tensor]:
    """
    Crops a batch of images to their original size, removing any padding

    Args:
        batch (Tensor): A batch of shape :math:`(N, C, H, W)` of images which may have been padded either by JPEG or to make them the same size
        sizes (Tensor): A tensor of shape :math:`(N, M)` where the height and width of image `i` respecively are stored at position `[i, -1]` and `[i, -2]`.

    Returns:
        Sequence of Tensors: A list of the cropped images, potentially all with different sizes.
    """
    return [batch[i, :, : int(sizes[i, -2]), : int(sizes[i, -1])] for i in range(len(batch))]
