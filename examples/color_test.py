import argparse

from PIL import Image
from torchvision.transforms.functional import to_pil_image, to_tensor

from torchjpeg.dct import to_rgb, to_ycbcr

parser = argparse.ArgumentParser("Tests the color conversion functions")
parser.add_argument("input", help="Input image, should be lossless")
args = parser.parse_args()

im = to_tensor(Image.open(args.input))

if im.shape[0] > 3:
    im = im[:3]

to_pil_image(to_rgb(to_ycbcr(im, data_range=1.0), data_range=1.0)).save("1_range.png")
to_pil_image(to_rgb(to_ycbcr(im * 255, data_range=255), data_range=255) / 255).save("255_range.png")
to_pil_image(to_rgb(to_ycbcr(im * 255, data_range=255) / 255, data_range=1.0)).save("255_1_range.png")
to_pil_image(to_rgb(to_ycbcr(im, data_range=1.0) * 255, data_range=255) / 255).save("1_255_range.png")
