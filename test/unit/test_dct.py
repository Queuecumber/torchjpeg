import torch

import torchjpeg.dct


def test_zigzag():
    """
    Tests torchjpeg.dct.zigzag by trying to zigzag the integers from [0, 63] in row major order. If it's
    working, then it should match the manual zigzag order.
    """
    # fmt: off
    zigzag_indices = torch.tensor([
         0,  1,  5,  6, 14, 15, 27, 28,
         2,  4,  7, 13, 16, 26, 29, 42,
         3,  8, 12, 17, 25, 30, 41, 43,
         9, 11, 18, 24, 31, 40, 44, 53,
        10, 19, 23, 32, 39, 45, 52, 54,
        20, 22, 33, 38, 46, 51, 55, 60,
        21, 34, 37, 47, 50, 56, 59, 61,
        35, 36, 48, 49, 57, 58, 62, 63 
    ]).float()
    # fmt: on

    assert torchjpeg.dct.zigzag(torch.arange(64).view(1, 8, 8).float()).squeeze().equal(zigzag_indices)
    assert torchjpeg.dct.zigzag(torch.arange(64).view(1, 1, 8, 8).float()).squeeze().equal(zigzag_indices)
