r"""
The :py:mod:`torchjpeg.quantization` package provides functions which quantize DCT coefficients. The IJG (libjpeg) quantization matrices
are included as part of this package as well as code which generates them from a scalar quality factor. Users can also provide their 
own quantization matrices. This implementation of the IJG quantization matrices only deals with the "baseline" setting where the 
maximum quantization value is 255. Functions in this module operate on single channel images since the channels are often quantized  
separately or may not be at the same resolution. 
"""


import torch
from torch import Tensor
from torch.nn.functional import upsample
from torchjpeg.dct import blockify, deblockify
from typing import Optional, Callable


def quantize(dct: Tensor, mat: Tensor, round: Callable[[Tensor], Tensor]=torch.round) -> Tensor:
    r"""
    Quantizes DCT coefficients.

    Args:
        dct (Tensor): DCT coefficients of shape :math:`(N, 1, H, W)`.
        mat: (Tensor): Quantization matrix of shape :math:`(1, 1, 8, 8)`.
        round: (Callable[[Tensor], Tensor]): Rounding function to use, defaults to :py:func:`torch.round`.

    Returns:
        Tensor: Quantized DCT coefficients.

    Note:
        DCT quantization is computed as 

        .. math::
            \widetilde{D}_{ij} = \left\lfloor \frac{D_{ij}}{Q_{ij}} \right\rceil

        For DCT coefficients :math:`D` and quantization matrix :math:`Q`.
    """
    dct_blocks = blockify(dct, 8)
    quantized_blocks = round(dct_blocks / mat)
    quantized = deblockify(quantized_blocks, (dct.shape[2], dct.shape[3]))
    return quantized


def dequantize(dct: Tensor, mat: Tensor) -> Tensor:
    r"""
    Dequantize DCT coefficients.

    Args:
        dct (Tensor): Quantized DCT coefficients of shape :math:`(N, 1, H, W)`.
        mat: (Tensor): Quantization matrix of shape :math:`(1, 1, 8, 8)`.

    Returns:
        Tensor: Quantized DCT coefficients.

    Note:
        DCT dequantization is computed as 

        .. math::
            D_{ij} = \widetilde{D}_{ij} \cdot Q_{ij}

        For quantized DCT coefficients :math:`\widetilde{D}` and quantization matrix :math:`Q`.
    """
    dct_blocks = blockify(dct, 8)
    dequantized_blocks = dct_blocks * mat
    dequantized = deblockify(dequantized_blocks, (dct.shape[2], dct.shape[3]))
    return dequantized

from . import ijg