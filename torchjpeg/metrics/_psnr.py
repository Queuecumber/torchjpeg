import torch


def psnr(target, input):
    mse = torch.nn.functional.mse_loss(input, target, reduction='none')
    mse = mse.view(mse.shape[0], -1).mean(1)
    return 10 * torch.log10(1 / mse)