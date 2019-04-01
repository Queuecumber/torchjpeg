import torch
import torchvision
import torchjpeg
import math
import numpy as np
from PIL import Image
import argparse


def deblockify(blocks, ch, size):
    bs = blocks.shape[0] // ch
    block_size = blocks.shape[2]

    blocks = blocks.reshape(bs * ch, -1, block_size**2)
    blocks = blocks.transpose(1, 2)
    blocks = torch.nn.functional.fold(blocks, output_size=size, kernel_size=block_size, stride=block_size)
    blocks = blocks.reshape(bs, ch, size[0], size[1])

    return blocks


def normalize(N):
    n = torch.ones((N, 1))
    n[0, 0] = 1 / math.sqrt(2)
    return (n @ n.t())


def harmonics(N):
    spatial = torch.arange(float(N)).reshape((N, 1))
    spectral = torch.arange(float(N)).reshape((1, N))

    spatial = 2 * spatial + 1
    spectral = (spectral * math.pi) / (2 * N)

    return torch.cos(spatial @ spectral)


def block_idct(coeff, device=None):
    N = coeff.shape[3]

    n = normalize(N)
    h = harmonics(N)

    if device is not None:
        n = n.to(device)
        h = h.to(device)

    im = (1 / math.sqrt(2 * N)) * (h @ (n * coeff) @ h.t())
    return im


def pixels_for_channel(c, q):
    dequantized = c.float() * q.float()

    s = block_idct(dequantized) + 128
    s = s.clamp(0, 255)
    s = s.view(1, s.shape[1] * s.shape[2], 8, 8)
    s = deblockify(s, 1, (c.shape[1] * 8, c.shape[2] * 8))
    s = s.squeeze(0).squeeze(0).numpy()

    return s


parser = argparse.ArgumentParser(
    'Tests the pytorch DCT quantizer by quantizing an input image'
)
parser.add_argument('input', help='Input image, should be lossless for best results')
parser.add_argument('output', help='Output image, should be lossless for best results')
parser.add_argument('quality', type=int, help='Output quality on the 0-100 scale')
args = parser.parse_args()

im = Image.open(args.input)
im_tensor = torchvision.transforms.functional.to_tensor(im)

dimensions, quantization, Y_coefficients, CbCr_coefficients = torchdctloader.quantize_at_quality(im_tensor, args.quality)

channels = dimensions.shape[0]

Y_spatial = pixels_for_channel(Y_coefficients, quantization[0])
Y_spatial = Image.fromarray(Y_spatial.astype(np.uint8), mode='L')


if channels > 1:
    Cb_spatial = pixels_for_channel(CbCr_coefficients[0].unsqueeze(0), quantization[1])
    Cr_spatial = pixels_for_channel(CbCr_coefficients[1].unsqueeze(0), quantization[2])

    Cb_spatial = Image.fromarray(Cb_spatial.astype(np.uint8), mode='L')
    Cr_spatial = Image.fromarray(Cr_spatial.astype(np.uint8), mode='L')

    dst_size = np.array(Y_spatial.size)
    Cb_spatial = Cb_spatial.resize(dst_size)
    Cr_spatial = Cr_spatial.resize(dst_size)

    YCbCr = Image.merge('YCbCr', [Y_spatial, Cb_spatial, Cr_spatial])
    spatial = YCbCr.convert('RGB')

else:
    spatial = Y_spatial

spatial.save(args.output)
