from pathlib import Path
from typing import Any, Callable, Optional, Sequence, Union

import torch
from PIL import Image


class UnlabeledImageFolder(torch.utils.data.Dataset):
    """
    Dataset loading a folder of unlabeled images recursively. The images are loaded using PIL and otherwise unchanged, add a transform to turn them into Tensors

    Args:
        path (Path): The path to load recursively from
        extensions (List[str]): The image extensions to look for
        transform: Any transform to apply to the images after loading them
    """

    def __init__(self, path: Union[str, Path], extensions: Sequence[str] = [".bmp", ".png", ".jpg", ".ppm", ".pgm"], transform: Optional[Callable] = None) -> None:
        # pylint: disable=dangerous-default-value
        if isinstance(path, str):
            path = Path(path)

        self.path = path
        self.images = list(filter(lambda p: p.suffix in extensions, path.glob("**/*")))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx) -> Any:
        im_path = self.images[idx]

        with open(im_path, "rb") as f:
            im = Image.open(f)
            im.load()

        if self.transform is not None:
            im = self.transform(im)

        return im
