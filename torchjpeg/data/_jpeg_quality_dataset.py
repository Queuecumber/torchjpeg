from torch.utils.data import Dataset
from ..dct import images_to_batch
from io import BytesIO
from PIL import Image
from torchvision.transforms.functional import to_tensor, to_pil_image
from torch.nn.functional import pad
from torch import Tensor, ceil
from typing import Tuple, List, Iterable, Union
from torch import randint
import torch

class JPEGQualityDataset(Dataset):
    def __init__(self, datasets: List[Iterable], quality_range: Union[int, Tuple[int, int]]) -> None:
        self.datasets = datasets
        self.qualities = quality_range

        self.len_images = sum(len(d) for d in self.datasets)

    def __len__(self):
        return self.len_images
        
    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, Tensor]:
        for d in self.datasets:
            if idx >= len(d):
                idx -= len(d)
            else:
                break

        uncompressed_image = to_tensor(d[idx])
        if isinstance(self.qualities, tuple):
            quality = randint(self.qualities[0], self.qualities[1] + 1, (1, 1, 1), dtype=torch.float)
        else:
            quality = Tensor([self.qualities]).view(1, 1, 1)

        s = Tensor(list(uncompressed_image.shape))
        p = (ceil(s / 16) * 16 - s).long()
        uncompressed_image = to_pil_image(pad(uncompressed_image.unsqueeze(0), [0, p[2], 0, p[1]], 'replicate').squeeze(0))

        with BytesIO() as f:
            uncompressed_image.save(f, format='jpeg', quality=int(quality[0,0,0]))
            f.seek(0)
            compressed_image = Image.open(f)
            compressed_image.load()

        compressed_image = images_to_batch(to_tensor(compressed_image).unsqueeze(0)).squeeze(0)
        uncompressed_image = images_to_batch(to_tensor(uncompressed_image).unsqueeze(0)).squeeze(0)

        return quality,  uncompressed_image, compressed_image
