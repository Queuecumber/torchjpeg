import torch
from torch import Tensor
from torch.nn.functional import upsample
from torchjpeg.dct import blockify, deblockify, batch_dct, batch_idct, block_idct, to_rgb
from .ijgquant import get_coefficients_for_quality
from typing import Optional, Callable


def quantize(dct: Tensor, mat: Tensor, round: Callable[[Tensor], Tensor]=torch.round) -> Tensor:
    dct_blocks = blockify(dct, 8)
    quantized_blocks = round(dct_blocks / mat)
    quantized = deblockify(quantized_blocks, (dct.shape[2], dct.shape[3]))
    return quantized


def dequantize(dct: Tensor, mat: Tensor) -> Tensor:
    dct_blocks = blockify(dct, 8)
    dequantized_blocks = dct_blocks * mat
    dequantized = deblockify(dequantized_blocks, (dct.shape[2], dct.shape[3]))
    return dequantized


def quantize_at_quality(dct_blocks: Tensor, quality: int, table: str='luma') -> Tensor:
    mat = get_coefficients_for_quality(quality, table=table)
    return quantize(dct_blocks, mat)


def dequantize_at_quality(dct_blocks: Tensor, quality: int, table: str='luma') -> Tensor:
    mat = get_coefficients_for_quality(quality, table=table)
    return dequantize(dct_blocks, mat)


def compress_coefficients(batch: Tensor, quality: int, table: str='luma') -> Tensor:
    batch = batch * 255 - 128
    d = batch_dct(batch)
    d = quantize_at_quality(d, quality, table=table)
    return d


def decompress_coefficients(batch: Tensor, quality: int, table: str='luma') -> Tensor:
    d = dequantize_at_quality(batch, quality, table=table)
    d = batch_idct(d)
    d = (d + 128) / 255
    return d