r"""
Provides access to low-level JPEG operations using libjpeg.
By using libjpeg directly, coefficients can be loaded or saved to JPEG files directly with needing to be recomputed.
In addtion to the C++ implemented low-level operations, two python convenience functions are exported that can decode the ressulting coefficients to pixels.
"""
from typing import Optional

import torch
from torch import Tensor
from torch.nn.functional import interpolate

from torchjpeg.dct import block_idct, deblockify, double_nn_dct, to_rgb

from ._codec_ops import *

__all__ = ["read_coefficients", "write_coefficients", "quantize_at_quality", "pixels_for_channel", "reconstruct_full_image"]


def pixels_for_channel(channel: Tensor, quantization: Tensor, crop: Optional[Tensor] = None) -> Tensor:
    r"""
    Converts a single channel of quantized DCT coefficients into pixels.

    Args
    ----------
    channel : torch.Tensor
        A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor of quantized DCT coefficients. 
    quantization : torch.Tensor
        An (8, 8) Tensor of the quantization matrix that was used to quantize :code:`channel`. 
    crop : torch.Tensor
        An optional (2) Tensor of containing the `$\left(H, W \right)$` original sizes of the image channel stored in :code:`channel`. The pixel result will be cropped to this size.

    Returns
    -------
    torch.Tensor
        A :math:`\left(H, W \right)` Tensor containing the pixel values of the channel in [0, 1]

    Note
    -----
    This function takes inputs in the same format as returned by :py:func:`read_coefficients` separated into a single channel.
    """
    dequantized = channel.float() * quantization.float()

    s = block_idct(dequantized) + 128
    s = s.view(1, 1, s.shape[1] * s.shape[2], 8, 8)
    s = deblockify(s, (channel.shape[1] * 8, channel.shape[2] * 8))
    s = s.squeeze()

    if crop is not None:
        s = s[: int(crop[0]), : int(crop[1])]

    return s


def reconstruct_full_image(y_coefficients: Tensor, quantization: Tensor, cbcr_coefficients: Optional[Tensor] = None, crop: Optional[Tensor] = None) -> Tensor:
    r"""
    Converts quantized DCT coefficients into an image.

    This function is designed to work on the output of :py:func:`read_coefficients` and py:func:`quantize_at_quality`. Note that the color channel coefficients
    will be upsampled by 2 as chroma subsampling is currently assumed. If the image is color, it will be converted from YCbCr to RGB. 

    Parameters
    ----------
    y_coefficients : torch.Tensor
        A :math:`\left(1, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor of quantized Y channel DCT coefficients.
    quantization : torch.Tensor
        A :math:`\left(C, 8, 8 \right)` Tensor of quantization matrices for each channel.
    cbcr_coefficients : Optional[torch.Tensor]
        A :math:`\left(2, \frac{H}{8}, \frac{W}{8}, 8, 8 \right)` Tensor of quantized color channel DCT coeffcients. 
    crop : Optional[torch.Tensor]
        A :math:`\left(C, 2 \right)` Tensor containing the :math:`\left(H, W \right)` dimensions of the image that produced the given  DCT coefficients, the pixel result will be cropped to this size.

    Returns
    -------
    torch.Tensor
        A :math:`\left(C, H, W \right)` Tensor containing the image pixels in pytorch format (normalized to [0, 1])    
    """
    y = pixels_for_channel(y_coefficients, quantization[0], crop[0] if crop is not None else None)

    if cbcr_coefficients is not None:
        cb = pixels_for_channel(cbcr_coefficients[0:1], quantization[1], crop[1] if crop is not None else None)
        cr = pixels_for_channel(cbcr_coefficients[1:2], quantization[2], crop[2] if crop is not None else None)

        cb = interpolate(cb.unsqueeze(0).unsqueeze(0), y.shape, mode="nearest")
        cr = interpolate(cr.unsqueeze(0).unsqueeze(0), y.shape, mode="nearest")

        out = torch.cat([y.unsqueeze(0).unsqueeze(0), cb, cr], dim=1)
        out = to_rgb(out).squeeze()
    else:
        out = y

    return out.clamp(0, 255) / 255.0
