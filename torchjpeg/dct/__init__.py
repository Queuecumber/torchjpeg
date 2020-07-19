r"""
The :code:`torchjpeg.dct` package provides utilities for performing forward and inverse discrete cosine transforms on images.
The dct routines are implemented in pytorch so they can be GPU accelerated and differentiated.
While the routines here are restricted to two dimensional signals, the block size is configurable e.g. the DCT does 
not need to be performed on only the :math:`8 \times 8` block size used by JPEG.
This package includes additional utilities for splitting images into non-overlapping blocks, performing fast 
color transforms on Tensors, and normalizing DCT coefficients as preparation for input to a CNN.
"""
from ._dct import *
from ._dct_nn import double_nn_dct, half_nn_dct
from ._dct_stats import DCTStats
from ._color import *

__all__ = [
    'blockify',
    'deblockify',
    'block_dct',
    'block_idct',
    'batch_dct',
    'batch_idct',
    'dct',
    'idct',
    'to_ycbcr',
    'to_rgb',
    'normalize',
    'denormalize',
    'batch_to_images',
    'images_to_batch',
    'double_nn_dct',
    'half_nn_dct',
    'DCTStats'
]