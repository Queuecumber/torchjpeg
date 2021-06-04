import numpy as np
import torch
from torch import Tensor


def blocking_effect_factor(im: Tensor) -> Tensor:
    r"""
    Computes the blocking effect factor (BEF) of an image as defined in [1].

    Blocking effect factor is used as part of :py:func:`psnrb` but can also be used
    as an objective measure of "blockiness".

    Args:
        im (Tensor): Image of shape :math:`(N, C, H, W)`.

    Returns:
        Tensor: The BEF for each image of shape :math:`(N)`.

    Note:
        [1] Tadala, Trinadh, and Sri E. Venkata Narayana. "A Novel PSNR-B Approach for Evaluating the Quality of De-blocked Images." (2012).
    """
    # This is a fairly complex formula we're implementing, it would be messy with fewer locals
    # pylint: disable=too-many-locals
    block_size = 8

    block_horizontal_positions = torch.arange(7, im.shape[3] - 1, 8)
    block_vertical_positions = torch.arange(7, im.shape[2] - 1, 8)

    horizontal_block_difference = ((im[:, :, :, block_horizontal_positions] - im[:, :, :, block_horizontal_positions + 1]) ** 2).sum(3).sum(2).sum(1)
    vertical_block_difference = ((im[:, :, block_vertical_positions, :] - im[:, :, block_vertical_positions + 1, :]) ** 2).sum(3).sum(2).sum(1)

    nonblock_horizontal_positions = np.setdiff1d(torch.arange(0, im.shape[3] - 1), block_horizontal_positions)
    nonblock_vertical_positions = np.setdiff1d(torch.arange(0, im.shape[2] - 1), block_vertical_positions)

    horizontal_nonblock_difference = ((im[:, :, :, nonblock_horizontal_positions] - im[:, :, :, nonblock_horizontal_positions + 1]) ** 2).sum(3).sum(2).sum(1)
    vertical_nonblock_difference = ((im[:, :, nonblock_vertical_positions, :] - im[:, :, nonblock_vertical_positions + 1, :]) ** 2).sum(3).sum(2).sum(1)

    n_boundary_horiz = im.shape[2] * (im.shape[3] // block_size - 1)
    n_boundary_vert = im.shape[3] * (im.shape[2] // block_size - 1)
    boundary_difference = (horizontal_block_difference + vertical_block_difference) / (n_boundary_horiz + n_boundary_vert)

    n_nonboundary_horiz = im.shape[2] * (im.shape[3] - 1) - n_boundary_horiz
    n_nonboundary_vert = im.shape[3] * (im.shape[2] - 1) - n_boundary_vert
    nonboundary_difference = (horizontal_nonblock_difference + vertical_nonblock_difference) / (n_nonboundary_horiz + n_nonboundary_vert)

    scaler = np.log2(block_size) / np.log2(min([im.shape[2], im.shape[3]]))
    bef = scaler * (boundary_difference - nonboundary_difference)

    bef[boundary_difference <= nonboundary_difference] = 0
    return bef


def psnrb(image: Tensor, target: Tensor) -> Tensor:
    r"""
    Computes the peak signal-to-noise ratio with blocking effect factor from [1].

    PSNR-B augments the PSNR measure by including the "blockiness" of the degraded image as a way to reduce
    the PSNR. For multichannel inputs, the PSNR-B is computed separately for each channel and then averaged.

    Args:
        image (Tensor): The input images of shape :math:`(N, C, H, W)`.
        target (Tensor): The target images of shape :math:`(N, C, H, W)`.

    Returns:
        Tensor: The PSNR-B of each image in the batch of shape :math:`(N)`.

    Warning:
        Unlike most metrics this is not symmetric and the order of the arguents is imporant. Blocking effect factor
        is only computed for the degraded image, so if the arguments are reversed, there will be very little difference
        between this and :py:func:`psnr`.

    Note:
        PSNR-B is computed as

        .. math::
            P(x, y) = 10 \log_{10}\left(\frac{1}{\text{MSE}(x, y) + \text{BEF}(x)}\right)

        [1] Tadala, Trinadh, and Sri E. Venkata Narayana. "A Novel PSNR-B Approach for Evaluating the Quality of De-blocked Images." (2012).
    """

    def channel_psnrb(image, target):
        mse = torch.nn.functional.mse_loss(image, target, reduction="none")
        bef = blocking_effect_factor(image)

        mse = mse.view(mse.shape[0], -1).mean(1)
        return 10 * torch.log10(1 / (mse + bef))

    total = torch.stack([channel_psnrb(image[:, c : c + 1, ...], target[:, c : c + 1, ...]) for c in range(image.shape[1])]).sum(0)
    return total / image.shape[1]
