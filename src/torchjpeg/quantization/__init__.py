r"""
The :py:mod:`torchjpeg.quantization` package provides functions which quantize DCT coefficients. The IJG (libjpeg) quantization matrices
are included as part of this package as well as code which generates them from a scalar quality factor. Users can also provide their 
own quantization matrices. This implementation of the IJG quantization matrices only deals with the "baseline" setting where the 
maximum quantization value is 255. Functions in this module operate on single channel images since the channels are often quantized  
separately or may not be at the same resolution. 
"""


from typing import Callable, Optional, Tuple

import torch
from torch import Tensor
from torch.nn.functional import upsample

from torchjpeg.dct import blockify, deblockify, double_nn_dct, half_nn_dct

from . import ijg
from ._quantize import dequantize, quantize


def quantize_multichannel(dct: Tensor, mat: Tensor, round_func: Callable[[Tensor], Tensor] = torch.round) -> Tuple[Tensor, Tensor, Tensor]:
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

    y_coefficients = quantize(y_coefficients, mat[:, 0:1, :, :], round_func=round_func,)
    cb_coefficients = quantize(cb_coefficients, mat[:, 1:2, :, :], round_func=round_func)
    cr_coefficients = quantize(cr_coefficients, mat[:, 1:2, :, :], round_func=round_func)

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


__all__ = ["quantize", "dequantize", "quantize_multichannel", "dequantize_multichannel"]
