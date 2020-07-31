import torch
from torch import Tensor


def to_ycbcr(x: Tensor) -> Tensor:
    r"""
    Converts a Tensor from RGB color space to YCbCr color space

    Parameters
    ----------
    x : Tensor
        The input Tensor holding an RGB image in [0, 255]. Can be in :math:`(N, C, H ,W)` or :math:`(C, H, W)` format.

    Returns
    -------
    Tensor
        The YCbCr result in [0, 255] of the same shape as the input.

    Note
    -----
    This function implements the "full range" conversion used by JPEG, e.g. it does **not** implement the ITU-R BT.601 standard which 
    many libraries (excluding PIL) use as the default definition of YCbCr. This conversion is given by:

    .. math::
        \begin{aligned}
        Y&=&0&+(0.299&\cdot R)&+(0.587&\cdot G)&+(0.114&\cdot B) \\
        C_{B}&=&128&-(0.168736&\cdot R)&-(0.331264&\cdot G)&+(0.5&\cdot B) \\
        C_{R}&=&128&+(0.5&\cdot R)&-(0.418688&\cdot G)&-(0.081312&\cdot B)
        \end{aligned}
    
    """
    # fmt: off
    ycbcr_from_rgb = torch.tensor([
        0.29900, 0.58700, 0.11400,
        -0.168735892, -0.331264108, 0.50000,
        0.50000, -0.418687589, -0.081312411
    ]).view(3, 3).transpose(0, 1)
    # fmt: on

    b = torch.tensor([0, 128, 128]).view(1, 3, 1, 1)

    if x.is_cuda:
        ycbcr_from_rgb = ycbcr_from_rgb.cuda()
        b = b.cuda()

    x = torch.einsum("cv,bcxy->bvxy", [ycbcr_from_rgb, x])
    x += b

    return x.contiguous()


def to_rgb(x: Tensor) -> Tensor:
    r"""
    Converts a Tensor from YCbCr color space to RGB color space

    Parameters
    ----------
    x : Tensor
        The input Tensor holding a YCbCr image in [0, 255]. Can be in :math:`(N, C, H ,W)` or :math:`(C, H, W)` format.

    Returns
    -------
    Tensor
        The RGB result in [0, 255] of the same shape as the input.

    Note
    -----
    This function expects the input to be "full range" conversion used by JPEG, e.g. it does **not** implement the ITU-R BT.601 standard which 
    many libraries (excluding PIL) use as the default definition of YCbCr. If the input came from this library or from PIL it should be fine.
    The conversion is given by:

    .. math::
        \begin{aligned}
        R&=&Y&&&+1.402&\cdot (C_{R}-128) \\
        G&=&Y&-0.344136&\cdot (C_{B}-128)&-0.714136&\cdot (C_{R}-128 ) \\
        B&=&Y&+1.772&\cdot (C_{B}-128)&
        \end{aligned}
    
    """
    # fmt: off
    rgb_from_ycbcr = torch.tensor([
        1, 0, 1.40200,
        1, -0.344136286, -0.714136286,
        1, 1.77200, 0
    ]).view(3, 3).transpose(0, 1)
    # fmt: on

    b = torch.tensor([0, 128, 128]).view(1, 3, 1, 1)

    if x.is_cuda:
        rgb_from_ycbcr = rgb_from_ycbcr.cuda()
        b = b.cuda()

    x -= b
    x = torch.einsum("cv,bcxy->bvxy", [rgb_from_ycbcr, x])

    return x.contiguous()
