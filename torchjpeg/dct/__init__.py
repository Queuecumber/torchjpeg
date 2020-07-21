r"""
The :code:`torchjpeg.dct` package provides utilities for performing forward and inverse discrete cosine transforms on images.
The dct routines are implemented in pytorch so they can be GPU accelerated and differentiated.
While the routines here are restricted to two dimensional signals, the block size is configurable e.g. the DCT does 
not need to be performed on only the :math:`8 \times 8` block size used by JPEG.
This package includes additional utilities for splitting images into non-overlapping blocks, performing fast 
color transforms on Tensors, and normalizing DCT coefficients as preparation for input to a CNN.
"""
import torch
import math
from torch import Tensor
from typing import Tuple, Optional
from torch.nn.functional import pad

from ._nn import double_nn_dct, half_nn_dct
from ._stats import Stats
from ._color import *

__all__ = [
    'blockify',
    'deblockify',
    'block_dct',
    'block_idct',
    'batch_dct',
    'batch_idct',
    'dct',
    'idct',
    'to_ycbcr',
    'to_rgb',
    'normalize',
    'denormalize',
    'batch_to_images',
    'images_to_batch',
    'double_nn_dct',
    'half_nn_dct',
    'Stats',
    'pad_to_block_multiple',
    'zigzag',
]


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

    im = im.view(bs * ch, 1, h, w)
    im = torch.nn.functional.unfold(im, kernel_size=size, stride=size)
    im = im.transpose(1, 2)
    im = im.view(bs, ch, -1, size, size)

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

    blocks = blocks.reshape(bs * ch, -1, block_size**2)
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=block_size, stride=block_size)
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks


def _normalize(N: Tensor) -> Tensor:
    r"""
    Computes the constant scale factor which makes the DCT orthonormal
    """
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return (n @ n.t())


def _harmonics(N: Tensor) -> Tensor:
    r"""
    Computes the cosine harmonics for the DCT transform
    """
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(blocks: Tensor) -> Tensor:
    r"""
    Computes the DCT of image blocks

    Args:
        blocks (Tensor): Non-overlapping blocks to perform the DCT on in :math:`(N, C, L, H, W)` format.
    
    Returns:
        Tensor: The DCT coefficients of each block in the same shape as the input.

    Note:
        The function computes the forward DCT on each block given by 

        .. math::

            D_{i,j}={\frac {1}{\sqrt{2N}}}\alpha (i)\alpha (j)\sum _{x=0}^{N}\sum _{y=0}^{N}I_{x,y}\cos \left[{\frac {(2x+1)i\pi }{2N}}\right]\cos \left[{\frac {(2y+1)j\pi }{2N}}\right]
        
        Where :math:`i,j` are the spatial frequency indices, :math:`N` is the block size and :math:`I` is the image with pixel positions :math:`x, y`. 
        
        :math:`\alpha` is a scale factor which ensures the transform is orthonormal given by 

        .. math::

            \alpha(u) = \begin{cases}{
                    \frac{1}{\sqrt{2}}} &{\text{if }}u=0 \\
                    1 &{\text{otherwise}}
                \end{cases}
        
        There is technically no restriction on the range of pixel values but to match JPEG it is recommended to use the range [-128, 127].
    """
    N = blocks.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if blocks.is_cuda:
        n = n.cuda()
        h = h.cuda()

    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ blocks @ h)

    return coeff


def block_idct(coeff: Tensor) -> Tensor:
    r"""
    Computes the inverse DCT of non-overlapping blocks

    Args:
        coeff (Tensor): The blockwise DCT coefficients in the format :math:`(N, C, L, H, W)`

    Returns:
        Tensor: The pixels for each block in the same format as the input. 

    Note:
        This function computes the inverse DCT given by 

        .. math::
            I_{x,y}={\frac {1}{\sqrt{2N}}}\sum _{i=0}^{N}\sum _{j=0}^{N}\alpha (i)\alpha (j)D_{i,j}\cos \left[{\frac {(2x+1)i\pi }{2N}}\right]\cos \left[{\frac {(2y+1)j\pi }{2N}}\right]

        See :py:func:`block_dct` for further details. 
    """
    N = coeff.shape[3]

    n = _normalize(N)
    h = _harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


def batch_dct(batch: Tensor) -> Tensor:
    r"""
    Computes the DCT of a batch of images. See :py:func:`block_dct` for more details. 
    This function takes care of splitting the images into blocks for the :py:func:`block_dct` and reconstructing
    the original shape of the input after the DCT.

    Args:
        batch (Tensor): A batch of images of format :math:`(N, C, H, W)`.

    Returns:
        Tensor: A batch of DCT coefficients of the same format as the input.

    Note:
        This fuction uses a block size of 8 to match the JPEG algorithm.
    """
    ch = batch.shape[1]
    size = (batch.shape[2], batch.shape[3])

    im_blocks = blockify(batch, 8)
    dct_blocks = block_dct(im_blocks)
    dct = deblockify(dct_blocks, size)

    return dct


def batch_idct(coeff: Tensor) -> Tensor:
    r"""
    Computes the inverse DCT of a batch of coefficients. See :py:func:`block_dct` for more details. 
    This function takes care of splitting the images into blocks for the :py:func:`block_idct` and reconstructing
    the original shape of the input after the inverse DCT.

    Args:
        batch (Tensor): A batch of coefficients of format :math:`(N, C, H, W)`.

    Returns:
        Tensor: A batch of images of the same format as the input.

    Note:
        This function uses a block size of 8 to match the JPEG algorithm.
    """
    ch = coeff.shape[1]
    size = (coeff.shape[2], coeff.shape[3])

    dct_blocks = blockify(coeff, 8)
    im_blocks = block_idct(dct_blocks)
    im = deblockify(im_blocks, size)

    return im


def dct(im: Tensor) -> Tensor:
    r"""
    Convenience function for taking the DCT of a single image

    Args:
        im (Tensor): A single image of format :math:`(C, H, W)`

    Returns: 
        Tensor: The DCT coefficients of the input in the same format.

    Note:
        This function simply expands the input in the batch dimension and then calls :py:func:`batch_dct` then removes
        the added batch dimension of the result.
    """
    return batch_dct(im.unsqueeze(0)).squeeze(0)


def idct(coeff: Tensor) -> Tensor:
    r"""
    Convenience function for taking the inverse InversDCT of a single image

    Args:
        im (Tensor): DCT coefficients of format :math:`(C, H, W)`

    Returns: 
        Tensor: The image pixels of the input in the same format.

    Note:
        This function simply expands the input in the batch dimension and then calls :py:func:`batch_idct` then removes
        the added batch dimension of the result.
    """
    return batch_idct(coeff.unsqueeze(0)).squeeze(0)


def normalize(dct: Tensor, stats: Stats, type: Optional[str] = None) -> Tensor:
    r"""
    Normalizes DCT coefficients using pre-computed stats

    This function wraps the :py:class:`DCTStats` class to allow easy normalization of multichannel images

    Args:
        dct (Tensor): The DCT coefficients to normalize in the format :math:`(N, C, H, W)`
        stats (:py:class:`DCTStats`): Precomputed DCT statistics
        type (Optional[str]): If the input coefficients are single channel, the Y channel is assumed, use this parameter to override that.

    Returns:
        Normalized DCT coefficients in the format :math:`(N, C, H, W)`
    """
    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

        t = ['y', 'cb', 'cr'][i] if type is None else type
        dct_blocks = stats.normalize(dct_blocks, type=t)

        ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

    return torch.cat(ch, dim=1)


def denormalize(dct: Tensor, stats: Stats, type: Optional[str] = None) -> Tensor:
    r"""
    Denormalizes DCT coefficients using pre-computed stats

    This function wraps the :py:class:`DCTStats` class to allow easy denormalization of multichannel images

    Args:
        dct (Tensor): The DCT normalized coefficients in the format :math:`(N, C, H, W)`
        stats (:py:class:`DCTStats`): Precomputed DCT statistics
        type (Optional[str]): If the input coefficients are single channel, the Y channel is assumed, use this parameter to override that.

    Returns:
        Denormalized DCT coefficients in the format :math:`(N, C, H, W)`
    """
    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

        t = ['y', 'cb', 'cr'][i] if type is None else type
        dct_blocks = stats.denormalize(dct_blocks, type=t)

        ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

    return torch.cat(ch, dim=1)


def batch_to_images(dct: Tensor, stats: Optional[Stats] = None, crop: Optional[Tensor] = None, type: Optional[str] = None) -> Tensor:
    r"""
    Converts a batch of DCT coefficients to a batch of images. 
    
    This high level convenience function wraps several operations. If stats are given, the coefficients are 
    assumed to have been channel-wise and frequency-wise normalized and are denormalized. The coefficients are tranformed to pixels and uncentered (converted from
    [-128, 127] to [0, 255]). If the input is multichannel, it is converted from YCbCr to RGB. The image is then optionally cropped to remove padding that may 
    have been added to make the coefficients. The output is then rescaled to [0, 1] to match pytorch conventions.

    Args:
        dct (Tensor): A batch of DCT coefficients in :math:`(N, C, H, W)` format.
        stats (Optional[:py:class:DCTStats]): Optional DCT per-channel and per-frequency statistics to denormalize the coefficients.
        crop (Optional[Tensor]): Optional cropping dimensions. If this tensor has more than a single dimension, only the last dimension is used.
        type (Optional[str]): One of 'Y', 'Cb', 'Cr'. Denormalization of a single channel input assumes 'Y' channel by default, use this paramter to override that.

    Returns:
        Tensor: The batch of images computed from the given coefficients.

    """
    if stats is not None:
        dct = denormalize(dct, stats, type=type)

    spatial = batch_idct(dct) + 128

    if spatial.shape[1] == 3:
        spatial = to_rgb(spatial)

    spatial = spatial.clamp(0, 255)
    spatial = spatial / 255

    if crop is not None:
        while len(crop.shape) > 1:
            crop = crop[0]

        cropY = crop[-2]
        cropX = crop[-1]

        spatial = spatial[:, :, :cropY, :cropX]

    return spatial


def images_to_batch(spatial: Tensor, stats: Optional[Stats] = None, type: Optional[str] = None) -> Tensor:
    r"""
    Converts a batch of images to a batch of DCT coefficients. 
    
    This high level convenience function wraps several operations. The input images are assumed to follow the pytorch convention of being in [0, 1] 
    and are rescaled to [0, 255]. If the images are multichannel, they are converted to YCbCr and then centered (in [-128, 127]). The 
    DCT is taken and if stats are given, the coefficients are normalized.

    Args:
        spatial (Tensor): A batch of images in :math:`(N, C, H, W)` format.
        stats (Optional[:py:class:DCTStats]): Optional DCT per-channel and per-frequency statistics to normalize the coefficients.
        type (Optional[str]): One of 'Y', 'Cb', 'Cr'. Normalization of a single channel input assumes 'Y' channel by default, use this paramter to override that.
        

    Returns:
        A batch of DCT coefficients computed from the input images.
    """
    spatial *= 255

    if spatial.shape[1] == 3:
        spatial = to_ycbcr(spatial)

    spatial -= 128

    frequency = batch_dct(spatial)
    if stats is not None:
        return normalize(frequency, stats, type=type)
    else:
        return frequency


def pad_to_block_multiple(im: Tensor, macroblock_size: int = 16) -> Tensor:
    r"""
    Pads an image to make it fit an even number of blocks

    Args:
        im (Tensor): the input image to pad, can be of shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.
        macroblock_size (int): The size of a macroblock, sometimes referred to as a minimum coded unit (MCU), default 16.

    Note:
        As in the JPEG standard, the padding is applied to the right and bottom edges and is replicate padding. Note that the default
        macroblock is of size 16 (meaning a :math:`16 \times 16` block). This ensures that if chroma subsampling is used on the color
        channels, after half-sizing they will still fit an even number of :math:`8 \times 8` blocks as required by the JPEG DCT.
    """
    s = torch.Tensor(list(im.shape))
    
    if len(s) != 4:
        im = im.unsqueeze(0)

    p = (torch.ceil(s / macroblock_size) * macroblock_size - s).long()
    im = pad(im, [0, p[-1], 0, p[-2]], 'replicate')

    if len(s) != 4:
        im = im.squeeze(0)

    return im


def zigzag(coefficients: Tensor) -> Tensor:
    r"""
    Vectorizes a DCT coefficients in JPEG zigzag order

    Args:
        coefficients (Tensor): DCT coefficients of shape :math:`(N, C, H, W)` or :math:`(C, H, W)`.
    
    Returns:
        Tensor: A batch of vectorized coefficients of shape :math:`(N, C, L, 64)` or :math:`(C, L, 64)`

    Note:
        For a visual representation of JPEG zigzag order see https://en.wikipedia.org/wiki/JPEG#/media/File:JPEG_ZigZag.svg.
    """
    assert len(coefficients.shape) in (3, 4)

    zigzag_indices = Tensor([
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 38, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 36, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63 
    ]).long()

    if len(coefficients.shape) == 3:
        c = coefficients.unsqueeze(0)
    else:
        c = coefficients

    c = blockify(c, 8)

    c = c.view(c.shape[0], c.shape[1], c.shape[2], 64)
    c[:, :, :, :] = c[:, :, :, zigzag_indices]

    if len(coefficients.shape) == 3:
        c = c.squeeze(0)

    return c