import argparse

import torchjpeg.codec

parser = argparse.ArgumentParser("Losslessly converts a JPEG to grayscale by dropping the Cb and Cr channels without requantizing")
parser.add_argument("input", help="Input image, must be a JPEG")
parser.add_argument("output", help="Output image, must be a JPEG")
args = parser.parse_args()

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.read_coefficients(args.input)
torchjpeg.codec.write_coefficients(args.output, dimensions, quantization, Y_coefficients)
