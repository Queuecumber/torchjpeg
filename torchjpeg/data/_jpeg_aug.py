from PIL import Image
from typing import Tuple
from io import BytesIO
from random import randint

class RandomJPEG(object):
    def __init__(self, quality_range: Tuple[int, int] = (0, 100)) -> None:
        self.quality_range = quality_range

    def __call__(self, input):
        q = randint(*self.quality_range)
        with BytesIO() as f:
            input.save(f, format='jpeg', quality=q)
            f.seek(0)
            output = Image.open(f)
            output.load()
        
        return output

    def __repr__(self):
        return f'{self.__class__.__name__} ({self.quality_range})'