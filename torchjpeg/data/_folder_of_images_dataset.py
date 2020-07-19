import torch
from glob import glob
import os
from PIL import Image


class FolderOfImagesDataset(torch.utils.data.Dataset):
    def __init__(self, path, patterns=['*.bmp', '*.png', '*.jpg', '*.ppm', '*.pgm'], transform=None):
        self.path = path
        self.images = [f for g in patterns for f in glob(os.path.join(path, g))]
        self.transform = transform

        im_path = self.images[0]

        with open(im_path, 'rb') as f:
            im = Image.open(f)
            im.load()

        self.channels = len(im.getbands())

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        im_path = self.images[idx]

        with open(im_path, 'rb') as f:
            im = Image.open(f)
            im.load()

        if self.transform is not None:
            im = self.transform(im)

        return im
