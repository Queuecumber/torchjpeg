import math

import torch
from torch import Tensor

from ._block import blockify, deblockify


def double_size_tensor() -> Tensor:
    """box resize, takes and 8 x 8 image and returns a 16 x 16 image"""
    op = torch.zeros((8, 8, 16, 16))
    for i in range(0, 8):
        for j in range(0, 8):
            for u in range(0, 16):
                for v in range(0, 16):
                    if i == u // 2 and j == v // 2:
                        op[i, j, u, v] = 1

    return op


def half_size_tensor() -> Tensor:
    """box resize, takes a 16 x 16 and returns an 8 x 8"""
    op = torch.zeros((16, 16, 8, 8))
    for i in range(0, 16):
        for j in range(0, 16):
            for u in range(0, 8):
                for v in range(0, 8):
                    if i == 2 * u and j == 2 * v:
                        op[i, j, u, v] = 1

    return op


def A(alpha) -> float:
    """DCT orthonormal scale factor"""
    if alpha == 0:
        return 1.0 / math.sqrt(2)

    return 1.0


def D() -> Tensor:
    """DCT tensor"""
    D_t = torch.zeros((8, 8, 8, 8))

    for i in range(8):
        for j in range(8):
            for alpha in range(8):
                for beta in range(8):
                    scale_a = A(alpha)
                    scale_b = A(beta)

                    coeff_x = math.cos(((2 * i + 1) * alpha * math.pi) / 16)
                    coeff_y = math.cos(((2 * j + 1) * beta * math.pi) / 16)

                    D_t[i, j, alpha, beta] = 0.25 * scale_a * scale_b * coeff_x * coeff_y
    return D_t


def reblock() -> Tensor:
    """reblocker, takes a 16 x 16 and returns 4 8 x 8 blocks"""
    B_t = torch.zeros((16, 16, 4, 8, 8))

    # This is deep but it really needs to be
    # pylint: disable=too-many-nested-blocks
    for s_x in range(16):
        for s_y in range(16):
            for n in range(4):
                for i in range(8):
                    for j in range(8):
                        x = n % 2
                        y = n // 2
                        if x * 8 + i == s_x and y * 8 + j == s_y:
                            B_t[s_x, s_y, n, i, j] = 1.0

    return B_t


def macroblock() -> Tensor:
    """Takes 4 x 8 x 8 and returns a 16 x 16"""
    B_t = torch.zeros((4, 8, 8, 16, 16))

    # 0 goes in top left
    for alpha in range(8):
        for beta in range(8):
            B_t[0, alpha, beta, alpha, beta] = 1

    # 1 goes in top right
    for alpha in range(8):
        for beta in range(8):
            B_t[1, alpha, beta, alpha + 8, beta] = 1

    # 2 goes in bottom left
    for alpha in range(8):
        for beta in range(8):
            B_t[2, alpha, beta, alpha, beta + 8] = 1

    # 3 goes in bottom right
    for alpha in range(8):
        for beta in range(8):
            B_t[3, alpha, beta, alpha + 8, beta + 8] = 1

    return B_t


resizer = double_size_tensor()
halfsizer = half_size_tensor()
dct = D()
reblocker = reblock()
macroblocker = macroblock()

# block doubler combines the following linear operations in order: inverse DCT, NN doubling, reshape to 4 x 8 x 8, DCT, reshape back to 16 x 16
block_doubler = torch.einsum("ijab,ijmn,mnzxy,xypq,zpqrw->abrw", dct, resizer, reblocker, dct, macroblocker)

# 16 x 16 -> 4 x 8 x 8 -> idct -> 16 x 16 -> resize -> dct
block_halver = torch.einsum("mnzab,ijab,zijrw,rwxy,xypq->mnpq", reblocker, dct, macroblocker, halfsizer, dct)


def double_nn_dct(input_dct: Tensor) -> Tensor:
    r"""
    DCT domain nearest neighbor doubling

    The function computes a 2x nearest neighbor upsampling on DCT coefficients without converting them to pixels.
    It is equivalent to the following procedure: IDCT -> 2x upsampling -> DCT

    Args:
        input_dct (Tensor): The input DCT coefficients in the format :math:`(N, C, H, W)`

    Returns:
        Tensor: The coefficients of the resized image, double the height and width of the input.
    """
    if input_dct.is_cuda:
        dop = block_doubler.cuda()
    else:
        dop = block_doubler

    dct_blocks = blockify(input_dct, 8)
    dct_doubled = torch.einsum("abrw,ncdab->ncdrw", [dop, dct_blocks])
    deblocked_doubled = deblockify(dct_doubled, (input_dct.shape[2] * 2, input_dct.shape[3] * 2))

    return deblocked_doubled


def half_nn_dct(input_dct: Tensor) -> Tensor:
    r"""
    DCT domain nearest neighbor half-sizing

    The function computes a 2x nearest neighbor downsampling on DCT coefficients without converting them to pixels.
    It is equivalent to the following procedure: IDCT -> 2x downsampling -> DCT

    Args:
        input_dct (Tensor): The input DCT coefficients in the format :math:`(N, C, H, W)`

    Returns:
        Tensor: The coefficients of the resized image, halg the height and width of the input.
    """
    if input_dct.is_cuda:
        dop = block_halver.cuda()
    else:
        dop = block_halver

    dct_blocks = blockify(input_dct, 16)
    dct_halved = torch.einsum("abrw,ncdab->ncdrw", [dop, dct_blocks])
    deblocked_halved = deblockify(dct_halved, (input_dct.shape[2] // 2, input_dct.shape[3] // 2))

    return deblocked_halved
