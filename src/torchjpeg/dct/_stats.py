import importlib.resources
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

    Pre-computed stats are available for color or grayscale images (pass "color" and "grayscale" respectively for
    the root argument), these stats were computed from the Flickr 2k dataset, a large corpus of high quality images and
    are suitable for general use.

    Args:
        root (:py:class:`pathlib.Path`, string, or literals "color", "grayscale"): The path to load the statistics from or "color" to use built in color stats or "grayscale" to use built in grayscale stats.
        normtype (str): Either "ms" for mean-variance normalization or "01" for zero-one normalization.
    """

    def __init__(self, root: Union[str, Path], normtype: str = "ms") -> None:
        self.type = normtype

        if root in ("color", "grayscale"):
            reader = importlib.resources.open_binary("torchjpeg.dct.stats", f"{root}.pt")
        else:
            if isinstance(root, str):
                root = Path(root)

            reader = root.open("rb")

        stats = torch.load(reader)

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
        m = self.mean[normtype].to(blocks.device)
        s = self.std[normtype].to(blocks.device)

        return (blocks - m) / s

    def _zero_one_f(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        m = -self.min[normtype].to(blocks.device)
        s = self.max[normtype] - self.min[normtype]
        s = s.to(blocks.device)

        return (blocks + m) / s

    def _mean_variance_r(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        s = self.std[normtype].to(blocks.device)
        m = self.mean[normtype].to(blocks.device)

        return blocks * s + m

    def _zero_one_r(self, blocks: Tensor, normtype: str = "y") -> Tensor:
        s = self.max[normtype] - self.min[normtype]
        s = s.to(blocks.device)
        m = -self.min[normtype].to(blocks.device)

        return blocks * s - m
