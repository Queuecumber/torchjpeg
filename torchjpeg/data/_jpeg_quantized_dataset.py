import torch
import torchvision
import torchjpeg
from ..dct import *
from ._color_transforms import YChannel


class JPEGQuantizedDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, qualities, dct_stats, quantization_stats, crop=True, grayscale=True, macroblock_size=2):
        self.datasets = datasets
        self.qualities = qualities
        self.dct_stats = dct_stats
        self.quantization_stats = quantization_stats
        self.macroblock_size = macroblock_size * 8.

        transform = []

        if crop:
            transform.append(torchvision.transforms.RandomCrop(256))

        transform.append(torchvision.transforms.ToTensor())

        if grayscale:
            transform.append(YChannel())

        self.transform = torchvision.transforms.Compose(transform)

        self.len_images = sum(len(d) for d in self.datasets)

    def __len__(self):
        return self.len_images * len(self.qualities)

    def __dequantize_channel(self, channel, quantization):
        dequantized_dct = channel.float() * quantization
        dequantized_dct = dequantized_dct.view(1, dequantized_dct.shape[1] * dequantized_dct.shape[2], 8, 8)
        dequantized_dct = deblockify(dequantized_dct, 1, (channel.shape[1] * 8, channel.shape[2] * 8))

        return dequantized_dct

    def __getitem__(self, idx):
        image_index = idx % self.len_images
        quality_index = idx // self.len_images

        for d in self.datasets:
            if image_index >= len(d):
                image_index -= len(d)
            else:
                break

        image = d[image_index]

        transformed = self.transform(image)
        quality = self.qualities[quality_index]

        s = torch.Tensor(list(transformed.shape))
        p = (torch.ceil(s / self.macroblock_size) * self.macroblock_size - s).long()
        transformed = torch.nn.functional.pad(transformed.unsqueeze(0), [0, p[2], 0, p[1]], 'replicate').squeeze(0)

        _, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.quantize_at_quality(transformed, quality)

        groundtruth_dct = images_to_batch(transformed.unsqueeze(0), self.dct_stats).squeeze(0)  # This *has* to be after quantization

        quantization = quantization.float()

        y_q = quantization[0]
        y_dequantized = self.__dequantize_channel(Y_coefficients, y_q)

        y_q = self.quantization_stats.forward(y_q, table='luma')

        if s[0] == 3:
            c_q = quantization[1]  # Assume same quantization for cb and cr
            
            cb_dequantized = self.__dequantize_channel(CbCr_coefficients[0].unsqueeze(0), c_q)
            cr_dequantized = self.__dequantize_channel(CbCr_coefficients[1].unsqueeze(0), c_q)

            y_dequantized = prepare_dct(y_dequantized, self.dct_stats, type='y')
            cb_dequantized = prepare_dct(cb_dequantized, self.dct_stats, type='cb')
            cr_dequantized = prepare_dct(cr_dequantized, self.dct_stats, type='cr')

            c_q = self.quantization_stats.forward(c_q, table='chroma')

            return y_dequantized.squeeze(0), cb_dequantized.squeeze(0), cr_dequantized.squeeze(0), y_q.unsqueeze(0), c_q.unsqueeze(0), groundtruth_dct, s.long()
        else:
            y_dequantized = prepare_dct(y_dequantized, self.dct_stats)

            return y_dequantized.squeeze(0), y_q.unsqueeze(0), groundtruth_dct, s.long()
