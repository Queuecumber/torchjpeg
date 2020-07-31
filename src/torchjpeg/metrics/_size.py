from io import BytesIO
from typing import Tuple, Union

import torch
from PIL import Image
from torch import Tensor
from torchvision.transforms.functional import to_pil_image, to_tensor


def size(image: Union[Tensor, Image.Image], **kwargs) -> Tuple[Tensor, Tensor]:
    r"""
    Computes the size in bytes of a JPEG

    Args:
        image (Tensor or PIL Image): The image to compress
        kwargs: Arguments to pass to the PIL JPEG compressor (like quality or quantization matrices)

    Returns
    -------
        Tensor
            A single element tensor containing the size in bytes of the image after JPEG compression
        Tensor
            The compressed image

    Warning:
        The output of this function is **not** differentiable. It compresses the image to memory and reads the size of 
        the resulting buffer.
    """
    if isinstance(image, Tensor):
        image = to_pil_image(image)

    with BytesIO() as f:
        image.save(f, "jpeg", **kwargs)
        f.seek(0)
        s = f.getbuffer().nbytes

        im = Image.open(f)
        im.load()
        im = to_tensor(im)

    return torch.tensor([s]), im
