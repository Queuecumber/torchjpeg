import torch
from torch import Tensor
from .codec_ops import *
from torchjpeg.dct import block_idct, deblockify, to_rgb, double_nn_dct
from torch.nn.functional import interpolate
from typing import Optional


def pixels_for_channel(channel: Tensor, q: Tensor, crop: Optional[Tensor] = None) -> Tensor:
    dequantized = channel.float() * q.float()

    s = block_idct(dequantized) + 128
    s = s.view(1, 1, s.shape[1] * s.shape[2], 8, 8)
    s = deblockify(s, (channel.shape[1] * 8, channel.shape[2] * 8))
    s = s.squeeze()

    if crop is not None:
        s = s[:crop[0], :crop[1]]

    return s


def reconstruct_full_image(y_coefficients: Tensor, quantization: Tensor, cbcr_coefficients: Optional[Tensor] = None, crop: Optional[Tensor] = None):
    y = pixels_for_channel(y_coefficients, quantization[0], crop[0] if crop is not None else None)

    if cbcr_coefficients is not None:
        cb = pixels_for_channel(cbcr_coefficients[0:1], quantization[1], crop[1] if crop is not None else None)
        cr = pixels_for_channel(cbcr_coefficients[1:2], quantization[2], crop[2] if crop is not None else None)

        cb = interpolate(cb.unsqueeze(0).unsqueeze(0), y.shape, mode='nearest') 
        cr = interpolate(cr.unsqueeze(0).unsqueeze(0), y.shape, mode='nearest')

        out = torch.cat([y.unsqueeze(0).unsqueeze(0), cb, cr], dim=1)
        out = to_rgb(out).squeeze()
    else:
        out = y

    return out.clamp(0, 255) / 255.