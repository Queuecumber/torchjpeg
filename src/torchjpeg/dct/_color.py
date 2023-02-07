import torch
from torch import Tensor


def to_ycbcr(x: Tensor, data_range: float = 255) -> Tensor:
    r"""
    Converts a Tensor from RGB color space to YCbCr color space

    Parameters
    ----------
    x : Tensor
        The input Tensor holding an RGB image in :math:`(\ldots, C, H ,W)` format (where :math:`\ldots` indicates an arbitrary number of dimensions).
    data_range : float
        The range of the input/output data. i.e., 255 indicates pixels in [0, 255], 1.0 indicates pixels in [0, 1]. Only 1.0 and 255 are supported.

    Returns
    -------
    Tensor
        The YCbCr result of the same shape as the input and with the same data range.

    Note
    -----
    This function implements the "full range" conversion used by JPEG, e.g. it does **not** implement the ITU-R BT.601 standard which
    many libraries (excluding PIL) use as the default definition of YCbCr. This conversion (for [0, 255]) is given by:

    .. math::
        \begin{aligned}
        Y&=&0&+(0.299&\cdot R)&+(0.587&\cdot G)&+(0.114&\cdot B) \\
        C_{B}&=&128&-(0.168736&\cdot R)&-(0.331264&\cdot G)&+(0.5&\cdot B) \\
        C_{R}&=&128&+(0.5&\cdot R)&-(0.418688&\cdot G)&-(0.081312&\cdot B)
        \end{aligned}

    """
    assert data_range in [1.0, 255]

    # fmt: off
    ycbcr_from_rgb = torch.tensor([
        0.29900, 0.58700, 0.11400,
        -0.168735892, -0.331264108, 0.50000,
        0.50000, -0.418687589, -0.081312411
    ],
    device=x.device).view(3, 3).transpose(0, 1)
    # fmt: on

    if data_range == 255:
        b = torch.tensor([0, 128, 128], device=x.device).view(3, 1, 1)
    else:
        b = torch.tensor([0, 0.5, 0.5], device=x.device).view(3, 1, 1)

    x = torch.einsum("cv,...cxy->...vxy", [ycbcr_from_rgb, x])
    x += b

    return x.contiguous()


def to_rgb(x: Tensor, data_range: float = 255) -> Tensor:
    r"""
    Converts a Tensor from YCbCr color space to RGB color space

    Parameters
    ----------
    x : Tensor
        The input Tensor holding a YCbCr image in :math:`(\ldots, C, H ,W)` format (where :math:`\ldots` indicates an arbitrary number of dimensions).
    data_range : float
        The range of the input/output data. i.e., 255 indicates pixels in [0, 255], 1.0 indicates pixels in [0, 1]. Only 1.0 and 255 are supported.

    Returns
    -------
    Tensor
        The RGB result of the same shape as the input and with the same data range.

    Note
    -----
    This function expects the input to be "full range" conversion used by JPEG, e.g. it does **not** implement the ITU-R BT.601 standard which
    many libraries (excluding PIL) use as the default definition of YCbCr. If the input came from this library or from PIL it should be fine.
    The conversion (for [0, 255]) is given by:

    .. math::
        \begin{aligned}
        R&=&Y&&&+1.402&\cdot (C_{R}-128) \\
        G&=&Y&-0.344136&\cdot (C_{B}-128)&-0.714136&\cdot (C_{R}-128 ) \\
        B&=&Y&+1.772&\cdot (C_{B}-128)&
        \end{aligned}

    """
    assert data_range in [1.0, 255]

    # fmt: off
    rgb_from_ycbcr = torch.tensor([
        1, 0, 1.40200,
        1, -0.344136286, -0.714136286,
        1, 1.77200, 0
    ],
    device=x.device).view(3, 3).transpose(0, 1)
    # fmt: on

    if data_range == 255:
        b = torch.tensor([-179.456, 135.458816, -226.816], device=x.device).view(3, 1, 1)
    else:
        b = torch.tensor([-0.70374902, 0.531211043, -0.88947451], device=x.device).view(3, 1, 1)

    x = torch.einsum("cv,...cxy->...vxy", [rgb_from_ycbcr, x])
    x += b

    return x.contiguous()
