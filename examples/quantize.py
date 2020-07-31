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
