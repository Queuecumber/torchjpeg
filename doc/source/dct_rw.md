# Reading and Writing DCT Coefficients

One of the main features of the `torchjpeg` library is functions related to reading and writing DCT coefficients from JPEG files. 
There are two major purposes to this functionality:
1. Reading coefficients is more efficient than reading pixels.
2. Reading coefficients constrains floating point error over reading pixels and transforming them back to DCT coefficients.

Similarly, if your project produces DCT coefficients, it is more desirable to write them directly after quantization. 

The {py:mod}`torchjpeg.codec` package provides an interface to libjpeg DCT reading and writing routines to make this possible. 

## Reading DCT Coefficients

Probably the most common use of this library is to read DCT coefficients. This is easy to do using the function {py:func}`torchjpeg.codec.read_coefficients` 
which takes a single argument: the path to the file to read. A simple example of this can be seen in the `examples/decompress.py` file, reproduced here:

``` {code-block} python
---
linenos: yes
emphasize-lines: '12'
---
import argparse

from torchvision.transforms.functional import to_pil_image

import torchjpeg.codec

parser = argparse.ArgumentParser("Tests the pytorch DCT loader by using it along with a custom DCT routine to decompress a JPEG")
parser.add_argument("input", help="Input image, must be a JPEG")
parser.add_argument("output", help="Output image, should be lossless for best results")
args = parser.parse_args()

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(args.input)
spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)

to_pil_image(spatial).save(args.output)
```

This script reads the coefficients from a given JPEG file, reconstructs the pixels, and saves them to another image. Note the highlighted
line 12, which uses {py:func}`torchjpeg.codec.read_coefficients`. The return value from this function is somewhat complex. 

The first return value gives the dimensions of the original file, *not* of the coefficients, as pytorch tensor. It is a $C \times 2$ tensor for a $C$ channel
image. This is necessary because often padding is added to the image to make it an even multiple of 8 blocks before the DCT is taken.
After decoding the coefficients, this size can be used to correctly crop the pixels to remove the padding. Note that this size is taken after chroma subsampling
(if applied) which is why a per-channel size is returned.

The second return value gives the quantization matrices used to quantize the coefficients as a pytorch tensor. It is a $C \times 8 \times 8$ tensor, since 
each channel is quantized independantly. Although libjpeg currently does not support separate Cb and Cr channel quantizations, it still stores 3 channel quanitzation
matrices and we return them as three channel here in case that changes in the future. 

The next return value is the quantized coefficients for the Y channel. Note that these are quantized coefficients, so they need to be dequantized using the quantization
matrix for the Y channel before they can be interpreted as pixels. 

The final return value is the Cb and Cr coefficients if any. Because there may have been chroma subsampling applied, we cannot return the Y Cb and Cr coefficients in a single 3 channel
tensor. If the image was grayscale the value will be `None`. 

To interpret each channel as pixels, follow roughly the following procedure:

$$
P_i = \text{IDCT}(C_i \times Q_i)
$$

where $C_i$ is the coeffcients for the $i$-th channel, $Q_i$ is the quantization matrix for the $i$-th channel, and $P_i$ is the pixels for the $i$-th channel. This operation is encapsulated in 
{py:func}`torchjpeg.codec.pixels_for_channel`. Doing this for all three channels as well as undoing chroma subsampling is encapsulated in {py:func}`torchjpeg.codec.reconstruct_full_image`.

```{tip}
The outputs from {py:func}`torchjpeg.codec.read_coefficients` are of type {py:data}`torch.short` (or equivilently {py:data}`torch.int16`) except the dimensions which are of type {py:data}`torch.int32`.
This is to match the output from libjpeg. If you want to do math of them, including if you want to do your own conversion to pixels, make sure to cast to {py:data}`torch.float` before using them.
```

## Writing DCT Coefficients

For a method which produces DCT coefficients, either on it's own or as a result of processing DCT coefficients, the function {py:func}`torchjpeg.codec.write_coefficients` can be used. An example
of this is provided in the `examples/grayscale.py` script, which losslessly converts a JPEG to grayscale. The script is reproduced below:

``` {code-block} python
---
linenos: yes
emphasize-lines: '11'
---
import argparse

import torchjpeg

parser = argparse.ArgumentParser("Losslessly converts a JPEG to grayscale by dropping the Cb and Cr channels without requantizing")
parser.add_argument("input", help="Input image, must be a JPEG")
parser.add_argument("output", help="Output image, must be a JPEG")
args = parser.parse_args()

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(args.input)
torchjpeg.codec.write_coefficients(args.output, dimensions, quantization, Y_coefficients)
```

Note that the first input to this function is the path to the JPEG file to create, and the remaining parameters are the same as the output from {py:func}`torchjpeg.codec.read_coefficients`. Also
note that the DCT inputs to this function must be quantized, and they must be the right type. For the dimensions, that is `torch.int` and for the coefficients and quantization matrices, it is `torch.short`.

Our lossless transcoding simply drops the Cb and Cr channel coefficients, but they can be provided as the last parameter to {py:func}`torchjpeg.codec.read_coefficients` to write color images.

```{tip}
The coefficients provided as input must be correctly quantized to ensure a decodable JPEG is created. You can use {py:mod}`torchjpeg.quantization` to help with this and you are not restricted to IJG (libjpeg)
quantization matrices, you can use whatever matrices you like and libjpeg will write them. In particular if your output is not intended to be quantized it is acceptable to round or truncate your coefficients and 
pass {py:func}`torch.ones` (equivalent to IJG quality 100) with the appropriate size and `dtype` as the quantization matrix. If you want to store floating point coefficients, JPEG is not what you want, use {py:func}`torch.save` instead.
```

## Bonus: Quantizing Pixels

As a bonus, the {py:mod}`torchjpeg.codec` package includes a function to quantize pixels and return the quantized coefficients using libjpeg. This is ideal if your intended application wants to ensure
that it is directly comparable to pixel based methods which use libjpeg (this includes MATLAB and PIL/Pillow). It is also fast, by using the libjpeg C implementation and performing the compression entirely
in memory. The function {py:func}`torchjpeg.codec.quantize_at_quality` implements this, an example is 
provided in the `examples/quantize.py` script and reproduced here:

```{code-block} python
---
linenos: yes
emphasize-lines: '20'
---
import argparse

from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

import torchjpeg.codec

parser = argparse.ArgumentParser("Tests the pytorch DCT quantizer by quantizing an input image")
parser.add_argument("input", help="Input image, should be lossless for best results")
parser.add_argument("output", help="Output image, should be lossless for best results")
parser.add_argument("quality", type=int, help="Output quality on the 0-100 scale")
args = parser.parse_args()

im = Image.open(args.input)
im_tensor = to_tensor(im)

if im_tensor.shape[0] > 3:
    im_tensor = im_tensor[:3]

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(im_tensor, args.quality)
spatial = torchjpeg.codec.reconstruct_full_image(Y_coefficients, quantization, CbCr_coefficients, dimensions)

to_pil_image(spatial).save(args.output)
```

The input image is a tensor in the standard  pytorch format, this means RGB (for color images) and pixels in the range [0, 1] of type `torch.float`. The second paramter
is an integer in [0, 100] specifying the quality (100 for high quality, 0 for low quality). The return value is exactly the same as {py:func}`torchjpeg.codec.read_coefficients`. For situations
requiring more flexibility where a perfect reproduction of libjpeg results (and speed) is less important, we provide the package {py:mod}`torchhjpeg.quantization` which implements general case 
JPEG quantization as well as the IJG quantization matrix from quality formula.