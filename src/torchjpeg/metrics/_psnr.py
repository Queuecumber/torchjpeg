import torch
from torch import Tensor


def psnr(image: Tensor, target: Tensor) -> Tensor:
    r"""
    Computes the peak signal-to-noise ratio on a batch of images.

    Args:
        image (Tensor): Input images in format :math:`(N, C, H, W)`. 
        target (Tensor): Target images in format :math:`(N, C, H, W)`.

    Returns:
        Tensor: PSNR for each image in the batch, of shape :math:`(N)`.

    Note:
        Peak signal-to-noise ratio is an inverse log scale of the mean squared error 
        measured in decibels. The formula used here is 

        .. math::

            P(x, y) = 10 \log_{10}\left(\frac{1}{\text{MSE}(x, y)}\right)
        
    """
    mse = torch.nn.functional.mse_loss(image, target, reduction="none")
    mse = mse.view(mse.shape[0], -1).mean(1)
    return 10 * torch.log10(1 / mse)
