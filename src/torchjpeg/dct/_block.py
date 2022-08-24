from typing import Tuple

import torch
import torch.nn.functional
from torch import Tensor


def blockify(im: Tensor, size: int) -> Tensor:
    r"""
    Breaks an image into non-overlapping blocks of equal size.

    Parameters
    ----------
    im : Tensor
        The image to break into blocks, must be in :math:`(N, C, H, W)` format.
    size : Tuple[int, int]
        The size of the blocks in :math:`(H, W)` format.

    Returns
    -------
    A tensor containing the non-overlappng blocks in :math:`(N, C, L, H, W)` format where :math:`L` is the
    number of non-overlapping blocks in the image channel indexed by :math:`(N, C)` and :math:`(H, W)` matches
    the block size.

    Note
    ----
    If the image does not split evenly into blocks of the given size, the result will have some overlap. It
    is the callers responsibility to pad the input to a multiple of the block size, no error will be thrown
    in this case.
    """
    bs = im.shape[0]
    ch = im.shape[1]
    h = im.shape[2]
    w = im.shape[3]

    im = im.reshape(bs * ch, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=(size, size), stride=(size, size))
    im = im.transpose(1, 2)
    im = im.reshape(bs, ch, -1, size, size)

    return im


def deblockify(blocks: Tensor, size: Tuple[int, int]) -> Tensor:
    r"""
    Reconstructs an image given non-overlapping blocks of equal size.

    Args:
        blocks (Tensor): The non-overlapping blocks in :math:`(N, C, L, H, W)` format.
        size: (Tuple[int, int]): The dimensions of the original image (e.g. the desired output)
            in :math:`(H, W)` format.

    Returns:
        The image in :math:`(N, C, H, W)` format.

    Note:
        If the blocks have some overlap, or if the output size cannot be constructed from the given number of non-overlapping
        blocks, this function will raise an exception unlike :py:func:`blockify`.

    """
    bs = blocks.shape[0]
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.reshape(bs * ch, -1, int(block_size**2))
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=(block_size, block_size), stride=(block_size, block_size))
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks
