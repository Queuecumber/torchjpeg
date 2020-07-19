import torch
from torch import Tensor

def to_ycbcr(x: Tensor) -> Tensor:
    ycbcr_from_rgb = torch.Tensor([
        0.29900, 0.58700, 0.11400,
        -0.168735892, -0.331264108, 0.50000,
        0.50000, -0.418687589, -0.081312411
    ]).view(3, 3).transpose(0, 1)

    b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1)

    if x.is_cuda:
        ycbcr_from_rgb = ycbcr_from_rgb.cuda()
        b = b.cuda()

    x = torch.einsum('cv,bcxy->bvxy', [ycbcr_from_rgb, x])
    x += b

    return x.contiguous()


def to_rgb(x: Tensor) -> Tensor:
    rgb_from_ycbcr = torch.Tensor([
        1, 0, 1.40200,
        1, -0.344136286, -0.714136286,
        1, 1.77200, 0
    ]).view(3, 3).transpose(0, 1)

    b = torch.Tensor([0, 128, 128]).view(1, 3, 1, 1)

    if x.is_cuda:
        rgb_from_ycbcr = rgb_from_ycbcr.cuda()
        b = b.cuda()

    x -= b
    x = torch.einsum('cv,bcxy->bvxy', [rgb_from_ycbcr, x])

    return x.contiguous()