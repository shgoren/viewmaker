import os
import copy
import getpass

import torchvision
from PIL import Image
import numpy as np
import random

import torch
import torch.utils.data as data
from torchvision import transforms, datasets
from viewmaker.src.datasets.root_paths import DATA_ROOTS


class FFHQ(data.Dataset):
    NUM_CLASSES = 1
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self,
            root=DATA_ROOTS['ffhq'],
            size=64,
            train=True,
            image_transforms=lambda x: x,
    ):
        super().__init__()
        self.size = size
        self.train = train
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize(self.size), image_transforms])
        self.dataset = datasets.ImageFolder(root, transform=self.transform)
        self.train_len = len(self.dataset) // 10 * 9
        if self.train:
            self.dataset.targets = self.dataset.targets[:self.train_len]
        else:
            self.dataset.targets = self.dataset.targets[self.train_len:]

    def __getitem__(self, index):
        # pick random number
        if self.train:
            offset = 0
        else:
            offset = self.train_len
        neg_index = np.random.choice(np.arange(self.__len__()))
        neg_index += offset
        index += offset
        img_data, label = self.dataset.__getitem__(index)
        img2_data, _ = self.dataset.__getitem__(index)
        neg_data, _ = self.dataset.__getitem__(neg_index)
        # build this wrapper such that we can return index
        data = [index, img_data.float(), img2_data.float(),
                neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        if self.train:
            return len(self.dataset) // 10 * 9
        else:
            return len(self.dataset) // 10


class FFHQ32(FFHQ):

    def __init__(
            self,
            root=DATA_ROOTS['ffhq'],
            train=True,
            image_transforms=lambda x: x,
    ):
        super().__init__(root=root, size=32, train=train, image_transforms=image_transforms)


class FFHQ64(FFHQ):

    def __init__(
            self,
            root=DATA_ROOTS['ffhq'],
            train=True,
            image_transforms=lambda x: x,
    ):
        super().__init__(root=root, size=64, train=train, image_transforms=image_transforms)


class FFHQ128(FFHQ):

    def __init__(
            self,
            root=DATA_ROOTS['ffhq'],
            train=True,
            image_transforms=lambda x: x,
    ):
        super().__init__(root=root, size=128, train=train, image_transforms=image_transforms)
