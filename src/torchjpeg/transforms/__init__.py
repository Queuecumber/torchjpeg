from io import BytesIO
from random import randint
from typing import Tuple

from PIL import Image


class YChannel:
    r"""
    Converts a tensor with a color image in [0, 1] to the Y channel using ITU-R BT.601 conversion

    Warning:
        This is **not** equivalent to the Y channel of a color image that would be used by JPEG, the result 
        is in [16, 240] following the ITU-R BT.601 standard before normalization. This is useful for certian JPEG artifact correction
        algorithms due to some questionable evaluation choices by that community. The result **is** normalized to :math:`\left[\frac{16}{255},\frac{240}{255}\right]`
        before being returned.
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        if tensor.shape[0] == 3:
            tensor = tensor * 255
            tensor = 16.0 + tensor[0, :, :] * 65.481 / 255.0 + tensor[1, :, :] * 128.553 / 255.0 + tensor[2, :, :] * 24.966 / 255.0
            tensor = tensor.unsqueeze(0).round() / 255.0

        return tensor

    def __repr__(self):
        return self.__class__.__name__ + "()"


class YCbCr:
    r"""
    Converts a PIL image to YCbCr color space

    Note:
        PIL follows the JPEG YCbCr color conversion giving a result in [0, 255]. 
    """

    def __init__(self):
        pass

    def __call__(self, pil_image):
        return pil_image.convert("YCbCr")

    def __repr__(self):
        return self.__class__.__name__ + "()"


class RandomJPEG:
    r"""
    Applies JPEG compression on a PIL at a random quality. 

    Args:
        quality_range (Tuple[int, int]): The quality range to choose from, inclusive on both ends. 
            An integer in this range will be chosen at random and will be used as the compression quality setting.
    """

    def __init__(self, quality_range: Tuple[int, int] = (0, 100)) -> None:
        self.quality_range = quality_range

    def __call__(self, pil_image):
        q = randint(*self.quality_range)
        with BytesIO() as f:
            pil_image.save(f, format="jpeg", quality=q)
            f.seek(0)
            output = Image.open(f)
            output.load()

        return output

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.quality_range})"
