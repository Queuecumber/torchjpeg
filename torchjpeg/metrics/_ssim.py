import torch


def ssim_single(target, input, device=None):
    C1 = 0.01**2
    C2 = 0.03**2

    filter = torch.ones(1, 1, 8, 8) / 64

    if device is not None:
        filter = filter.to(device)

    mu_i = torch.nn.functional.conv2d(input, filter)
    mu_t = torch.nn.functional.conv2d(target, filter)

    var_i = torch.nn.functional.conv2d(input**2, filter) - mu_i**2
    var_t = torch.nn.functional.conv2d(target**2, filter) - mu_t**2
    cov_it = torch.nn.functional.conv2d(target*input, filter) - mu_i * mu_t

    ssim_blocks = ((2 * mu_i * mu_t + C1) * (2 * cov_it + C2)) / ((mu_i**2 + mu_t**2 + C1) * (var_i + var_t + C2))
    return ssim_blocks.view(input.shape[0], -1).mean(1)


def ssim(target, input, device=None):
    total = 0
    for c in range(target.shape[1]):
        total += ssim_single(target[:, c:c+1, :, :], input[:, c:c+1, :, :], device)

    return total / target.shape[1]