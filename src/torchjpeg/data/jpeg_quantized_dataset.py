from random import randrange
from typing import Any, Mapping, Optional, Sequence, Tuple, Union

import torch
from torch.functional import Tensor
from torch.utils.data import Dataset

from torchjpeg.codec import quantize_at_quality  # type: ignore # pylint: disable=no-name-in-module
from torchjpeg.dct import Stats, deblockify, images_to_batch, normalize, pad_to_block_multiple
from torchjpeg.quantization.ijg import quantization_max

from .image_list import ImageList


def _dequantize_channel(channel, quantization):
    dequantized_dct = channel.float() * quantization
    dequantized_dct = dequantized_dct.view(1, 1, dequantized_dct.shape[1] * dequantized_dct.shape[2], 8, 8)
    dequantized_dct = deblockify(dequantized_dct, (channel.shape[1] * 8, channel.shape[2] * 8))

    return dequantized_dct


def _prep_coefficients(quantization: Tensor, Y_coefficients: Tensor, CbCr_coefficients: Tensor, stats: Stats):
    quantization = quantization.float()
    y_q = quantization[0]

    y_dequantized = _dequantize_channel(Y_coefficients, y_q)
    y_q = y_q / quantization_max
    y_dequantized = normalize(y_dequantized, stats, channel="y")

    if CbCr_coefficients is not None:
        c_q = quantization[1]  # Assume same quantization for cb and cr

        cb_dequantized = _dequantize_channel(CbCr_coefficients[0:1], c_q)
        cr_dequantized = _dequantize_channel(CbCr_coefficients[1:2], c_q)

        c_q = c_q / quantization_max

        cb_dequantized = normalize(cb_dequantized, stats, channel="cb")
        cr_dequantized = normalize(cr_dequantized, stats, channel="cr")

        cbcr_dequantized = torch.cat([cb_dequantized, cr_dequantized], dim=1)
    else:
        cbcr_dequantized = torch.empty(0)
        c_q = torch.empty(0)

    return y_dequantized, cbcr_dequantized, y_q, c_q


class JPEGQuantizedDataset(Dataset):
    r"""
    Wraps an arbitrary image dataset to return JPEG quantized versions of the image. The amount quantization is defined using IJG quality settings.
    If the underlying dataset returns a sequence, the first element of the sequence is taken the be the image which is quantized and the remaining
    elements are returned as the last element of the batch. If the underlying dataset returns a mapping, set `image_key` to the key of the image
    to be quantized. The original dictionary, including the image before quantization, will be returned as the last element of the batch.

    Since the primary return values are all DCT coefficients, padding will be added to the images to make them an even multiple of the MCU. Following JPEG
    conventions this is replicate padding added to the bottom and right edges. The original size of the image is returned so that the images can be
    correctly cropped after processing.

    The format returned by this dataset is:

    Y Channel Coefficients, CbCr Coefficients, Y Quantization Matrix, CbCr Quantization Matrix, Pre-quantization YCbCr Coefficients, Original Image Size, Optional rest of the batch from the underlying dataset

    If the image is grayscale, the CbCr coefficients and quantization matrix will be an empty tensor, if the underlying dataset returns an image with no additional data, the final return value will be an empty tensor.
    This is to avoid issues with the default collate function, an empty tensor is one initialized with 0 size using :py:func:`torch.empty`. It is detectable as `tensor.numel() == 0`.

    Args:
        data (:py:class:`torch.utils.data.Dataset`): The dataset to wrap
        quality (int, tuple of two or three ints): The quality range (min 0 max 100) to draw from, inclusive on both ends. If this is a single integer, only that quality is used, if it's three integers, the last one defines a step size.
        stats (:py:clas:`torchjpeg.dct.Stats`): Statstics to use for per-frequency per-channel coefficient normalization
        mcu (int): The size of the minimum coded unit, use 16 for 4:2:0 chroma subsampling.
        image_key (optional str): The key to use to extract the image from a dataset which returns a mapping.
        deterministic_quality (bool): False by default, set to True to include the quality range in the dataset size. In other words, the length of this dataset will be `len(quality_range) * len(dataset)` and all the qualities in the range will
            be represented for every image by interating this dataset.

    Warning:
        The images returned from this dataset may be of differing sizes, use the static :py:func:`torchjpeg.data.JPEGQuantizedDataset.collate` to collate them into a batch with padding. Use :py:func:`torchjpeg.data.crop_batch` to
        crop them back to the correct sizes (this will also remove JPEG padding).
    """

    def __init__(self, dataset: Dataset, quality: Union[int, Sequence[int]], stats: Stats, mcu: int = 16, image_key: Optional[str] = None, deterministic_quality: bool = False) -> None:
        # pylint: disable=too-many-arguments
        assert (isinstance(quality, Sequence) and len(quality) > 0) or isinstance(quality, int)

        if isinstance(quality, int):
            quality = (quality, quality + 1, 1)
        elif isinstance(quality, Sequence):
            if len(quality) == 1:
                quality = (quality[0], quality[0] + 1, 1)
            elif len(quality) == 2:
                quality = (quality[0], quality[1] + 1, 1)
            elif len(quality) > 2:
                quality = (quality[0], quality[1], quality[2])

        self.dataset = dataset
        self.quality = quality
        self.stats = stats
        self.mcu = mcu
        self.image_key = image_key
        self.deterministic_quality = deterministic_quality

    def __len__(self) -> int:
        if self.deterministic_quality:
            return len(self.dataset) * len(range(*self.quality))  # type: ignore

        return len(self.dataset)  # type: ignore

    def __getitem__(self, idx) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Any]:
        # pylint: disable=too-many-locals
        if not self.deterministic_quality:
            image = self.dataset[idx]
            quality = randrange(*self.quality)
        else:
            image_idx = idx % len(self.dataset)  # type: ignore
            quality_idx = idx // len(self.dataset)  # type: ignore

            image = self.dataset[image_idx]
            quality = quality_idx * self.quality[2] + self.quality[0]

        labels: Any = torch.empty(0)

        if isinstance(image, Sequence):
            labels = image[1:]
            image = image[0]
        elif isinstance(image, Mapping):
            labels = image
            image = image[self.image_key]

        s = torch.as_tensor(image.shape)
        image = pad_to_block_multiple(image)

        (
            _,
            quantization,
            Y_coefficients,
            CbCr_coefficients,
        ) = quantize_at_quality(image, quality)

        groundtruth_dct = images_to_batch(image.unsqueeze(0), self.stats).squeeze(0)  # This *has* to be after quantization

        y_dequantized, cbcr_dequantized, y_q, c_q = _prep_coefficients(quantization, Y_coefficients, CbCr_coefficients, self.stats)

        return (
            y_dequantized.squeeze(0),
            cbcr_dequantized.squeeze(0),
            y_q.unsqueeze(0),
            c_q.unsqueeze(0),
            groundtruth_dct,
            s.long(),
            labels,
        )

    @staticmethod
    def collate(batch_list: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Any]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Any]:
        """
        Custom collate function which works for return values from this dataset. Adds padding to the images so that they can be stored in a single tensor

        Args:
            batch_list: Output from this dataset

        Returns:
            Batch with each input collated into single tensors
        """
        y_coefs = []
        cbcr_coefs = []
        gt_coefs = []

        yqs = torch.stack([b[2] for b in batch_list])
        cqs = torch.stack([b[3] for b in batch_list])
        sizes = torch.stack([b[5] for b in batch_list])

        labels = [b[6] for b in batch_list]

        for b in batch_list:
            y_coefs.append(b[0])
            cbcr_coefs.append(b[1])
            gt_coefs.append(b[4])

        y_coefs_t = ImageList.from_tensors(y_coefs).tensor
        cbcr_coefs_t = ImageList.from_tensors(cbcr_coefs).tensor
        gt_coefs_t = ImageList.from_tensors(gt_coefs).tensor

        return y_coefs_t, cbcr_coefs_t, yqs, cqs, gt_coefs_t, sizes, labels
