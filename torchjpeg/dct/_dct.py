import torch
import math
from torch import Tensor
from ._dct_stats import DCTStats
from ._color import to_rgb, to_ycbcr
from typing import Tuple, Optional


def blockify(im: Tensor, size: Tuple[int, int]) -> Tensor:
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
    bs = blocks.shape[0] 
    ch = blocks.shape[1]
    block_size = blocks.shape[3]

    blocks = blocks.reshape(bs * ch, -1, block_size**2)
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=block_size, stride=block_size)
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks


def normalize(N: Tensor) -> Tensor:
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return (n @ n.t())


def harmonics(N: Tensor) -> Tensor:
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_dct(im: Tensor) -> Tensor:
    N = im.shape[3]

    n = normalize(N)
    h = harmonics(N)

    if im.is_cuda:
        n = n.cuda()
        h = h.cuda()

    coeff = (1 / math.sqrt(2 * N)) * n * (h.t() @ im @ h)

    return coeff


def block_idct(coeff: Tensor) -> Tensor:
    N = coeff.shape[3]

    n = normalize(N)
    h = harmonics(N)

    if coeff.is_cuda:
        n = n.cuda()
        h = h.cuda()

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


def batch_dct(im: Tensor) -> Tensor:
    ch = im.shape[1]
    size = (im.shape[2], im.shape[3])

    im_blocks = blockify(im, 8)
    dct_blocks = block_dct(im_blocks)
    dct = deblockify(dct_blocks, size)

    return dct


def batch_idct(dct: Tensor) -> Tensor:
    ch = dct.shape[1]
    size = (dct.shape[2], dct.shape[3])

    dct_blocks = blockify(dct, 8)
    im_blocks = block_idct(dct_blocks)
    im = deblockify(im_blocks, size)

    return im


def dct(im: Tensor) -> Tensor:
    return batch_dct(im.unsqueeze(0)).squeeze(0)


def idct(im: Tensor) -> Tensor:
    return batch_idct(im.unsqueeze(0)).squeeze(0)


def prepare_dct(dct: Tensor, stats: DCTStats, type: str = None) -> Tensor:
    # TODO fix this for new blockify format
    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

        t = ['y', 'cb', 'cr'][i] if type is None else type
        dct_blocks = stats.forward(dct_blocks, type=t)

        ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

    return torch.cat(ch, dim=1)


def unprepare_dct(dct: Tensor, stats: DCTStats, type: str = None) -> Tensor:
    # TODO fix this for new blockify format
    ch = []

    for i in range(dct.shape[1]):
        dct_blocks = blockify(dct[:, i:(i+1), :, :], 8)

        t = ['y', 'cb', 'cr'][i] if type is None else type
        dct_blocks = stats.backward(dct_blocks, type=t)

        ch.append(deblockify(dct_blocks, 1, dct.shape[2:]))

    return torch.cat(ch, dim=1)


def batch_to_images(dct: Tensor, stats: Optional[DCTStats] = None, scale_freq: bool = True, crop: bool = None, type: Optional[str] = None) -> Tensor:
    if scale_freq and stats is not None:
        dct = unprepare_dct(dct, stats, type=type)

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


def images_to_batch(spatial: Tensor, stats: Optional[DCTStats] = None, type: Optional[str] = None) -> Tensor:
    spatial *= 255

    if spatial.shape[1] == 3:
        spatial = to_ycbcr(spatial)

    spatial -= 128

    frequency = batch_dct(spatial)
    if stats is not None:
        return prepare_dct(frequency, stats, type=type)
    else:
        return frequency
