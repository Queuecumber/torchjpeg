import torchjpeg
import argparse
from PIL import Image
from torchvision.transforms.functional import to_tensor


parser = argparse.ArgumentParser(
    'Tests the pytorch DCT loader by reading and image, quantizing its pixels, and writing the DCT coefficients to a JPEG'
)
parser.add_argument('input', help='Input image, should be lossless')
parser.add_argument('output', help='Output image, must be a JPEG')
parser.add_argument('quality', type=int, help='Output quality on the 0-100 scale')
args = parser.parse_args()

im = to_tensor(Image.open(args.input))

if im.shape[0] > 3:
    im = im[:3]

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.codec.quantize_at_quality(im, args.quality)
torchjpeg.codec.write_coefficients(args.output, dimensions, quantization, Y_coefficients, CbCr_coefficients)
