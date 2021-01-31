from pathlib import Path
from typing import Union

import torch
from torch import Tensor


class Stats:
    r"""
    This class holds pre-computed per-channel and per-frequency DCT coefficient stats.

    The stats are loaded from a file, this can be written using :py:func:`torch.save`. 
    The file should contain a single dictionary with string keys containing channel names.
    The value of each entry should be a dictionary with the keys: "mean, variance, min, and max" 
    with the corresponding statistics as Tensors.

    Args:
        root (:py:class:`pathlib.Path`): The path to load the statistics from.
        normtype (str): Either "ms" for mean-variance normalization or "01" for zero-one normalization.
    """

    def __init__(self, root: Union[str, Path], normtype: str = "ms") -> None:
        self.type = normtype

        if isinstance(root, str):
            root = Path(root)

        stats = torch.load(root.open("rb"))

        self.mean = {x: stats[x]["mean"].view(1, 1, 8, 8) for x in stats.keys()}
        self.variance = {x: stats[x]["variance"].view(1, 1, 8, 8) for x in stats.keys()}
        self.std = {x: torch.sqrt(self.variance[x]) for x in stats.keys()}

        self.min = {x: stats[x]["min"].view(1, 1, 8, 8) for x in stats.keys()}
        self.max = {x: stats[x]["max"].view(1, 1, 8, 8) for x in stats.keys()}

    def normalize(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        r"""
        Normalizes blocks of coefficients.

        Args:
            blocks (Tensor): a Tensor containing blocks of DCT coefficients in the format :math:`(N, C, L, H, W)`.
            normtype (str): Which channel to normalize, "y" by default.

        Returns:
            Tensor: The normalized coefficients.
        """
        if self.type == "ms":
            return self._mean_variance_f(blocks, normtype)
        if self.type == "01":
            return self._zero_one_f(blocks, normtype)

        raise NotImplementedError(f"Unknown norm type {normtype}, must be 01 or ms")

    def denormalize(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        r"""
        Denormalizes blocks of coefficients.

        Args:
            blocks (Tensor): a Tensor containing blocks of normalized DCT coefficients in the format :math:`(N, C, L, H, W)`.
            normtype (str): Which channel to denormalize, "y" by default.

        Returns:
            Tensor: The denormalized coefficients.
        """
        if self.type == "ms":
            return self._mean_variance_r(blocks, normtype)
        if self.type == "01":
            return self._zero_one_r(blocks, normtype)

        raise NotImplementedError(f"Unknown norm type {normtype}, must be 01 or ms")

    def _mean_variance_f(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        m = self.mean[normtype]
        s = self.std[normtype]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()

        return (blocks - m) / s

    def _zero_one_f(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        m = -self.min[normtype]
        s = self.max[normtype] - self.min[normtype]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()

        return (blocks + m) / s

    def _mean_variance_r(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        s = self.std[normtype]
        m = self.mean[normtype]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()

        return blocks * s + m

    def _zero_one_r(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        s = self.max[normtype] - self.min[normtype]
        m = -self.min[normtype]

        if blocks.is_cuda:
            m = m.cuda()
            s = s.cuda()

        return blocks * s - m
