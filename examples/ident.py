import argparse

import torchjpeg.codec

parser = argparse.ArgumentParser("Tests the pytorch DCT loader by reading quantized DCT coefficients from a JPEG then writing them unchanged")
parser.add_argument("input", help="Input image, must be a JPEG")
parser.add_argument("output", help="Output image, must be a JPEG")
args = parser.parse_args()

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(args.input)
torchjpeg.codec.write_coefficients(args.output, dimensions, quantization, Y_coefficients, CbCr_coefficients)
