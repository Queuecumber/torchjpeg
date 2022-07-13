import argparse

from PIL import Image
from torchvision.transforms.functional import to_tensor

import torchjpeg.codec

parser = argparse.ArgumentParser("Tests the pytorch DCT loader by reading and image, quantizing its pixels, and writing the DCT coefficients to a JPEG")
parser.add_argument("input", help="Input image, should be lossless")
parser.add_argument("output", help="Output image, must be a JPEG")
parser.add_argument("quality", type=int, help="Output quality on the 0-100 scale")
parser.add_argument("color_samp_factor_vertical", type=int, nargs="?", default=2, help="Vertical chroma subsampling factor. Defaults to 2.")
parser.add_argument("color_samp_factor_horizontal", type=int, nargs="?", default=2, help="Horizontal chroma subsampling factor. Defaults to 2.")
args = parser.parse_args()

im = to_tensor(Image.open(args.input))

if im.shape[0] > 3:
    im = im[:3]

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(im, args.quality, args.color_samp_factor_vertical, args.color_samp_factor_horizontal)
torchjpeg.codec.write_coefficients(args.output, dimensions, quantization, Y_coefficients, CbCr_coefficients)
