import torch
from torch import Tensor


def ssim_single(
    image: Tensor,
    target: Tensor,
) -> Tensor:
    r"""
    Computes SSIM for a single channel
    """
    C1 = 0.01**2
    C2 = 0.03**2

    avg_filter = torch.ones(1, 1, 8, 8, device=image.device) / 64

    mu_i = torch.nn.functional.conv2d(image, avg_filter)
    mu_t = torch.nn.functional.conv2d(target, avg_filter)

    var_i = torch.nn.functional.conv2d(image**2, avg_filter) - mu_i**2
    var_t = torch.nn.functional.conv2d(target**2, avg_filter) - mu_t**2
    cov_it = torch.nn.functional.conv2d(target * image, avg_filter) - mu_i * mu_t

    ssim_blocks = ((2 * mu_i * mu_t + C1) * (2 * cov_it + C2)) / ((mu_i**2 + mu_t**2 + C1) * (var_i + var_t + C2))
    return ssim_blocks.view(image.shape[0], -1).mean(1)


def ssim(image: Tensor, target: Tensor) -> Tensor:
    r"""
    Computes the structural similarity index of two images as defined in [1].

    Args:
        image (Tensor): The input images of shape :math:`(N, C, H, W)`.
        target (Tensor): The target images of shape :math:`(N, C, H, W)`.

    Returns:
        Tensor: The SSIM of each image of shape :math:`(N)`.

    Note:
        This function uses an :math:`8 \times 8` uniform averaging window used in JPEG evaluation tasks instead of the :math:`11 \times 11` gaussian window
        used in the original paper and by default in other SSIM implementations.

        [1] Wang, Zhou, et al. "Image quality assessment: from error visibility to structural similarity." IEEE transactions on image processing 13.4 (2004): 600-612.
    """
    total = torch.stack([ssim_single(image[:, c : c + 1, :, :], target[:, c : c + 1, :, :]) for c in range(target.shape[1])]).sum(0)
    return total / target.shape[1]
