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
from torchjpeg.dct import blockify, deblockify, double_nn_dct, half_nn_dct
from typing import Optional, Callable, Tuple


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
    quantized_blocks = round(dct_blocks / mat.unsqueeze(2))
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
    dequantized_blocks = dct_blocks * mat.unsqueeze(2)
    dequantized = deblockify(dequantized_blocks, (dct.shape[2], dct.shape[3]))
    return dequantized


def quantize_multichannel(dct: Tensor, mat: Tensor, round: Callable[[Tensor], Tensor]=torch.round) -> Tuple[Tensor, Tensor, Tensor]:
    r"""
    Quantizes a three channel image of DCT coefficients.

    Args:
        dct (Tensor): DCT coefficients of shape :math:`(N, 3, H, W)`.
        mat (Tensor): Quantization matrix of shape :math:`(1, 2, 8, 8)`.
        round: (Callable[[Tensor], Tensor]): Rounding function to use, defaults to :py:func:`torch.round`.
    
    Returns
    -------
        Tensor
            Y channel coefficients of shape :math:`(N, 1, H, W)`.
        Tensor
            Cb channel coefficients of shape :math:`\left(N, 1, \frac{H}{2}, \frac{W}{2}\right)`.
        Tensor
            Cr channel coefficients of shape :math:`\left(N, 1, \frac{H}{2}, \frac{W}{2}\right)`.
        
    Note:
        This function performs chroma subsampling
    """
    y_coefficients = dct[:, 0:1, :, :]
    cb_coefficients = dct[:, 1:2, :, :]
    cr_coefficients = dct[:, 2:3, :, :]

    cb_coefficients = half_nn_dct(cb_coefficients)
    cr_coefficients = half_nn_dct(cr_coefficients)

    y_coefficients = quantize(y_coefficients, mat[:, 0:1, :, :])
    cb_coefficients = quantize(cb_coefficients, mat[:, 1:2, :, :])
    cr_coefficients = quantize(cr_coefficients, mat[:, 1:2, :, :])

    return y_coefficients, cb_coefficients, cr_coefficients


def dequantize_multichannel(y: Tensor, cb: Tensor, cr: Tensor, mat: Tensor) -> Tensor:
    r"""
    Dequantizes a three channel image.

    Args:
        y (Tensor): Quantized Y channel coefficients of shape :math:`(N, 1, H, W)`.
        cb (Tensor): Quantized  Cb channel coefficients of shape :math:`\left(N, 1, \frac{H}{2}, \frac{W}{2}\right)`.
        cr (Tensor): Quantized Cr channel coefficients of shape :math:`\left(N, 1, \frac{H}{2}, \frac{W}{2}\right)`.

    Returns:
        Tensor: A three channel image of DCT coefficients.

    Note:
        This function assumes chroma subsampling.
    """
    y_coefficients = dequantize(y, mat[:, 0:1, :, :])
    cb_coefficients = dequantize(cb, mat[:, 1:2, :, :])
    cr_coefficients = dequantize(cr, mat[:, 1:2, :, :])

    cb_coefficients = double_nn_dct(cb_coefficients)
    cr_coefficients = double_nn_dct(cr_coefficients)

    coefficients = torch.cat([y_coefficients, cb_coefficients, cr_coefficients], dim=1)
    return coefficients


from . import ijg