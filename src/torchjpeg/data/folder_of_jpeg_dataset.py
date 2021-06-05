from pathlib import Path
from typing import Sequence, Tuple, Union

import torch
from torch import Tensor

import torchjpeg.codec
from torchjpeg.dct import Stats

from .image_list import ImageList
from .jpeg_quantized_dataset import _prep_coefficients


class FolderOfJpegDataset(torch.utils.data.Dataset):
    """
    Loads coefficents from a folder of JPEG without any labels. For each image, it returns the format of :py:func:`torchjpeg.codec.read_coefficients`. The
    images must be actualy JPEG files (stored as JPEGs) for this to work. The relative path to the JPEG file will be returned along with the coefficients. The
    coefficients themselves are not guaranteed to be the same size, use the collate function to collate these into a batched Tensor by adding padding.

    Args:
        path (Path): The path to load images from
        stats (Stats): DCT stats to use to normalize the coefficients
        extensions (List[str]): The JPEG file extensions to search for
    """

    def __init__(self, path: Union[str, Path], stats: Stats, extensions: Sequence[str] = [".jpg", ".jpeg", ".JPEG"]):
        # pylint: disable=dangerous-default-value
        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.stats = stats

        if path.is_dir():
            self.images = list(filter(lambda p: p.suffix in extensions, path.glob("**/*")))
        else:
            self.images = [path]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]

        dim, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(str(image))  # type: ignore

        y_dequantized, cbcr_dequantized, y_q, c_q = _prep_coefficients(quantization, Y_coefficients, CbCr_coefficients, self.stats)

        return y_dequantized.squeeze(0), cbcr_dequantized.squeeze(0), y_q.unsqueeze(0), c_q.unsqueeze(0), image.relative_to(self.path), dim[0]

    @staticmethod
    def collate(batch_list: Sequence[Tuple[Tensor, Tensor, Tensor, Tensor, Path, Tensor]]) -> Tuple[Tensor, Tensor, Tensor, Tensor, Sequence[Path], Tensor]:
        """
        Custom collate function which works for return values from this dataset. Adds padding to the images so that they can be stored in a single tensor

        Args:
            batch_list: Output from this dataset

        Returns:
            Batch with each input collated into single tensors
        """
        y_coefs = []
        cbcr_coefs = []

        yqs = torch.stack([b[2] for b in batch_list])
        cqs = torch.stack([b[3] for b in batch_list])
        sizes = torch.stack([b[5] for b in batch_list])

        paths = [b[4] for b in batch_list]

        for b in batch_list:
            y_coefs.append(b[0])
            cbcr_coefs.append(b[1])

        y_coefs_t = ImageList.from_tensors(y_coefs).tensor
        cbcr_coefs_t = ImageList.from_tensors(cbcr_coefs).tensor

        return y_coefs_t, cbcr_coefs_t, yqs, cqs, paths, sizes
