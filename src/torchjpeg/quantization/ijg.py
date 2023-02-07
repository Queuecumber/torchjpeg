r"""
:py:mod:`torchjpeg.quantization.ijg` provides functions which match the Independent JPEG Group's libjpeg quantization method.
"""
import torch
from torch import Tensor

from torchjpeg.dct import batch_dct, batch_idct

from ._quantize import dequantize, quantize

# Don't mess with matrix formatting
# fmt: off
luma_quant_matrix = torch.tensor([
    16,  11,  10,  16,  24,  40,  51,  61,
    12,  12,  14,  19,  26,  58,  60,  55,
    14,  13,  16,  24,  40,  57,  69,  56,
    14,  17,  22,  29,  51,  87,  80,  62,
    18,  22,  37,  56,  68, 109, 103,  77,
    24,  35,  55,  64,  81, 104, 113,  92,
    49,  64,  78,  87, 103, 121, 120, 101,
    72,  92,  95,  98, 112, 100, 103,  99
])
# fmt: on

# Don't mess with matrix formatting
# fmt: off
chroma_quant_matrix = torch.tensor([
    17,  18,  24,  47,  99,  99,  99,  99,
    18,  21,  26,  66,  99,  99,  99,  99,
    24,  26,  56,  99,  99,  99,  99,  99,
    47,  66,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99,
    99,  99,  99,  99,  99,  99,  99,  99
])
# fmt: on

quantization_max = 255.0


def qualities_to_scale_factors(qualities: Tensor) -> Tensor:
    r"""
    Converts a batch of qualities in [0, 100] to a batch of scale factors suitable for scaling one of the IJG reference quantization matrices.

    Args:
        qualities (Tensor): A single dimensional batch of qualities.

    Returns:
        Tensor: A single dimensional batch of scale factors.
    """
    qualities = qualities.clone()
    qualities[qualities <= 0] = 1
    qualities[qualities > 100] = 100

    indices_0_50 = qualities < 50
    indices_50_100 = qualities >= 50

    qualities[indices_0_50] = 5000 // qualities[indices_0_50]
    qualities[indices_50_100] = torch.trunc(200 - qualities[indices_50_100] * 2)

    return qualities


def scale_quantization_matrices(scale_factor: Tensor, table: str = "luma") -> Tensor:
    r"""
    Scales one of the IJG reference quantization matrices.

    Args:
        scale_factor (Tensor): A batch of :math:`N` scale factors.
        table (str): A string indicating the table to use, either "luma" or "chroma"

    Returns:
        Tensor: A batch of quantization matrices of shape :math:`(N, 64)`.
    """
    if table == "luma":
        t = luma_quant_matrix
    elif table == "chroma":
        t = chroma_quant_matrix

    t = t.to(scale_factor.device)

    t = t.unsqueeze(0)
    scale_factor = scale_factor.unsqueeze(1)

    mat = (t * scale_factor + 50) // 100
    mat[mat <= 0] = 1
    mat[mat > 255] = 255

    return mat


def get_coefficients_for_qualities(quality: Tensor, table: str = "luma") -> Tensor:
    r"""
    Gets IJG quantization matrices for a given batch of qualities.

    Args:
        quality (Tensor): A batch of qualities of shape :math:`(N)`
        table (str): A string indicating the table to use, either "luma" or "chroma"

    """
    scaler = qualities_to_scale_factors(quality)
    mat = scale_quantization_matrices(scaler, table=table)
    return mat.view(-1, 1, 8, 8)


def quantize_at_quality(dct: Tensor, quality: int, table: str = "luma") -> Tensor:
    r"""
    Quantizes using a scalar quality instead of a quantization matrix. Uses IJG quantization matrices.

    Args:
        dct (Tensor): DCT coefficients of shape :math:`(N, 1, H ,W)`.
        quality (int): A scalar in [0, 100] specifying the desired quality.
        table (str): One of "luma" or "chroma" to choose the desired set of tables.

    Returns:
        Tensor: Quantized DCT coefficients.
    """
    mat = get_coefficients_for_qualities(torch.tensor([quality]), table=table)
    return quantize(dct, mat)


def dequantize_at_quality(dct: Tensor, quality: int, table: str = "luma") -> Tensor:
    r"""
    Dequantizes using a scalar quality instead of a quantization matrix. uses IJG quantization matrices.

    Args:
        dct (Tensor): Quantized DCT coefficients of shape :math:`(N, 1, H, W)`.
        quality (int): A scalar in [0, 100] specifying the quality that the coefficients were quantized at.
        table (str): One of "luma" or "chroma" to choose the desired set of tables.
    """
    mat = get_coefficients_for_qualities(torch.tensor([quality]), table=table)
    return dequantize(dct, mat)


def compress_coefficients(batch: Tensor, quality: int, table: str = "luma") -> Tensor:
    r"""
    A high level function that takes a batch of pixels in [0, 1] and returns quantized DCT coefficients.

    Args:
        batch (Tensor): A batch of images to quantize of shape `(N, 1, H, W)`.
        quality (int): A scalar quality in [0, 100] specifying the desired quality.
        table (str): One of "luma" or "chroma" to choose the desired set of tables.

    Returns:
        Tensor: A batch of quantized DCT coefficients.
    """
    batch = batch * 255 - 128
    d = batch_dct(batch)
    d = quantize_at_quality(d, quality, table=table)
    return d


def decompress_coefficients(batch: Tensor, quality: int, table: str = "luma") -> Tensor:
    r"""
    A high level function that converts quantized DCT coefficients to pixels.

    Args:
        batch (Tensor): A batch of quantized DCT coefficients of shape `(N, 1, H, W)`.
        quality (int): A scalar quality in [0, 100] specifying the quality the coefficients were quantized at.
        table (str): One of "luma" or "chroma" to choose the desired set of tables.

    Returns:
        Tensor: A batch of image pixels.
    """
    d = dequantize_at_quality(batch, quality, table=table)
    d = batch_idct(d)
    d = (d + 128) / 255
    return d
