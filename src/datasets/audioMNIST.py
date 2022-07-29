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
from collections import Counter

class AudioMNIST(data.Dataset):
    NUM_CLASSES = 10
    NUM_CHANNELS = 3
    FILTER_SIZE = 32
    MULTI_LABEL = False

    def __init__(
            self,
            root=DATA_ROOTS['audioMNIST'],
            size=64,
            train=True,
            image_transforms=lambda x: x,
            validation_split=0.1
    ):
        super().__init__()
        self.size = size
        self.train = train
        if image_transforms:
            self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((self.size,self.size)), image_transforms])
        else:
            self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((self.size,self.size)),torchvision.transforms.ToTensor()])
        self.dataset = datasets.ImageFolder(root, transform=self.transform)
        
        # val-train split
        np_labels = np.array(self.dataset.targets)
        counter = Counter(np_labels)
        self.db_subset_idx = []
        for lbl in counter:
            if self.train:
                self.db_subset_idx.extend(np.arange(0,len(np_labels))[np_labels==lbl][:int((1-validation_split)*counter[lbl])])
            else:
                self.db_subset_idx.extend(np.arange(0,len(np_labels))[np_labels==lbl][int((1-validation_split)*counter[lbl]):])

        self.dataset.targets = [ self.dataset.targets[i] for i in self.db_subset_idx ] #update targets for validations

        
        

    def __getitem__(self, index):
        # pick random number
        neg_index = np.random.choice(np.arange(self.__len__()))
        img_data, label = self.dataset.__getitem__(self.db_subset_idx[index])
        img2_data, _ = self.dataset.__getitem__(self.db_subset_idx[index])
        neg_data, _ = self.dataset.__getitem__(self.db_subset_idx[neg_index])
        # build this wrapper such that we can return index
        data = [index, img_data.float(), img2_data.float(),neg_data.float(), label]
        return tuple(data)

    def __len__(self):
        return len(self.db_subset_idx)
