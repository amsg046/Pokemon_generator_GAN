# -*- coding: utf-8 -*-
"""
Created on Sun Jan  9 11:20:46 2022

@author: Amar Singh
"""

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import torch

data_dir = "data/Abstract_gallery/Abstract_gallery"

# Image Parameters
image_px = 64
batch_size = 128
stats = (.5, .5, .5), (.5, .5, .5)

train_ds = ImageFolder(data_dir, transform=T.Compose([
    T.resize(image_px),
    T.CenterCrop(image_px),
    T.ToTensor(),
    T.Normalize(*stats)
    ]))

train_dl = DataLoader(train_ds, batch_size, shuffle=True)
