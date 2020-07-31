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
