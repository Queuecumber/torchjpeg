# Metrics and Loss Functions

The {py:mod}`torchjpeg.metrics` package provides useful metrics for evaluating JPEGs. In general, these metrics are differentiable and can be used as loss functions. Note that {py:func}`torchjpeg.metrics.size` is the only
metric which is not differentiable. It returns the size in bytes of a JPEG compressed image given the image pixels.

## JPEG Quality Evaluation

We provide three metrics for evaluating the visual quality of JPEG images: {py:func}`torchjpeg.metrics.psnr`, {py:func}`torchjpeg.metrics.psnrb`, and {py:func}`torchjpeg.metrics.ssim`. While these metrics are good attempts at 
quantifying the visual quality of an image, they are known to be imperfect. However, reporting all three of these metrics consistently is a good way to show that a particular method improves visual quality with respect to another method.

```{note}
For a more objective measure of image quality, it may be necessary to carry out a user study or rely on a metric such as the [FID score](https://arxiv.org/abs/1706.08500).
```

PSNR and PSNR-B are both measured in decibels and SSIM is unitless. To compute the quality of a JPEG image, code like the following can be used:

```{code-block} python
---
linenos: yes
emphasize-lines: '10,11,12'
---
from PIL import Image
from torchvision.transforms.functional import to_tensor

from torchjpeg.metrics import psnr, psnrb, ssim


image = to_tensor(Image.open('input.png'))
jpeg = to_tensor(Image.open('input.jpg'))

p = psnr(jpeg, image)
b = psnrb(jpeg, image)
s = ssim(jpeg, image)

print(f'PSNR: {p}db, PSNR-B: {b}db, SSIM: {s}')
```

For PSNR-B, the order of the arguments is extremely important. The JPEG must be passed in the first argument because the blocking-effect-factor is only computed on the degraded image. Passing the arguments
in reverse order will result in a very low BEF and therefore an artificially good score on PSNR-B. We have seen people make this mistake before. The BEF can be computed without PSNR-B using the function {py:func}`torchjpeg.metrics.blocking_effect_factor`. This takes a single image as input, so it is considered an "objective measure of blockiness".

For SSIM, we have hard coded some JPEG specific settings. These are based on the evaluation procedure of [ARCNN](http://mmlab.ie.cuhk.edu.hk/projects/ARCNN.html) a seminal work in JPEG artifact correction. Instead of 
the customary $11 \times 11$ gaussian window, it uses an $8 \times 8$ uniform averaging window. For a general use SSIM library, we recommend [pytorch-msssim](https://github.com/VainF/pytorch-msssim) which includes multi-scale
SSIM.

## Loss Functions

Since the quality metrics are differentiable they can be used as loss functions. There is one major caveat to this: they measure quality so the objective should be maximized and not minimized.

```{warning}
In general, we do not recommend using 
PSNR as a loss function, since it should be equivalent to minimizing the $l_2$ error of the image. PSNR-B has a similar issue, it may be a better idea to minimize BEF ({py:func}`torchjpeg.metrics.blocking_effect_factor`), the "objective measure of blockiness" used by PNSR-B.
```

For SSIM, there are two ways to construct a loss function:

```{code-block} python
---
linenos: yes
---

from torchjpeg.metrics import ssim

def ssim_loss_a(jpeg, target):
    return -ssim(jpeg, target)

def ssim_loss_b(jpeg, target):
    return 1 - ssim(jpeg, target)
```

In other words you can either minimize the negative SSIM or minimize 1 - ssim, since SSIM is in [0, 1]. The first option is slightly simpler while the second is more interpretable, but the result (gradient) should be the same.

## Size Metric

The {py:func}`torchjpeg.metrics.size` metric gives the size in bytes of a JPEG as it would be stored on disk. It takes as input an image, either a PIL image or a pytorch tensor of shape $C \times H \times W$ (pixels in [0, 1]) and
compresses it using PIL. The function also accepts `kwargs` to pass to PIL. You can use these kwargs to pass quality or custom quantization matrices. The return value is a single dimensional tensor containing the size in bytes of the
JPEG as it would be stored on disk, and a tensor containing the decompressed pixels.

To use the size metric with a quality:

```{code-block} python
---
linenos: yes
emphasize-lines: '9'
---
from PIL import Image
from torchvision.transforms.functional import to_tensor

from torchjpeg.metrics import size


im = to_tensor(Image.open('input.png'))

sizes = [size(im, quality=q)[0] for q in range(0, 101)]
print(sizes)
```

This code prints the size of an image for all 101 JPEG quality levels. 

To use the size metric with a quantization matrix, pass two 64 dimensional lists of integers made from flattening the quantization matrices in row-major order:

```{code-block} python
---
linenos: yes
emphasize-lines: '12'
---
from PIL import Image
from torchvision.transforms.functional import to_tensor

from torchjpeg.metrics import size


im = to_tensor(Image.open('input.png'))

qm_l = [1] * 64
qm_c = [1] * 64

size_q100, _ = size(im, qtables=[qm_l, qm_c])
print(size_q100)
```

Which prints the size of an image at quality 100 using the quanitzation matrix for quality 100 explicitly (all ones).
