import torch
from glob import glob
import os
from PIL import Image
import torchjpeg
from ..dct import *
from ..quantization import *


class FolderOfJpegDataset(torch.utils.data.Dataset):
    def __init__(self, path, dct_stats, patterns=['**/*.jpg']):
        self.path = path
        self.images = [f for g in patterns for f in glob(os.path.join(path, g))]
        self.dct_stats = dct_stats
        
    def __len__(self):
        return len(self.images)

    def __dequantize_channel(self, channel, quantization):
        dequantized_dct = channel.float() * quantization
        dequantized_dct = dequantized_dct.view(1, dequantized_dct.shape[1] * dequantized_dct.shape[2], 8, 8)
        dequantized_dct = deblockify(dequantized_dct, 1, (channel.shape[1] * 8, channel.shape[2] * 8))

        return dequantized_dct

    def __getitem__(self, idx):
        image = self.images[idx]

        dim, quantization, Y_coefficients, CbCr_coefficients = torchjpeg.read_coefficients(image)
        quantization = quantization.float()

        y_q = quantization[0]
        y_dequantized = self.__dequantize_channel(Y_coefficients, y_q)

        y_q /= quantization_max

        if CbCr_coefficients is not None:
            c_q = quantization[1]  # Assume same quantization for cb and cr
            
            cb_dequantized = self.__dequantize_channel(CbCr_coefficients[0].unsqueeze(0), c_q)
            cr_dequantized = self.__dequantize_channel(CbCr_coefficients[1].unsqueeze(0), c_q)

            y_dequantized = prepare_dct(y_dequantized, self.dct_stats, type='y')
            cb_dequantized = prepare_dct(cb_dequantized, self.dct_stats, type='cb')
            cr_dequantized = prepare_dct(cr_dequantized, self.dct_stats, type='cr')

            c_q /= quantization_max

            return y_dequantized.squeeze(0), cb_dequantized.squeeze(0), cr_dequantized.squeeze(0), y_q.unsqueeze(0), c_q.unsqueeze(0), image, dim
        else:
            y_dequantized = prepare_dct(y_dequantized, self.dct_stats)

            return y_dequantized.squeeze(0), y_q.unsqueeze(0), image, dim
