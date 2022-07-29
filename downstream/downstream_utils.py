import argparse
import json
import os

import numpy as np
import tensorflow as tf
import torch
import torchvision.utils
import wandb
from dotmap import DotMap

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.layers import MaxPool2D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.initializers import RandomNormal, Constant
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Activation, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import optimizers
# import tensorflow_addons as tfa

# Normalize images
from viewmaker.src.systems.image_systems import TransferViewMakerSystem, PretrainViewMakerSystemDisc, PretrainViewMakerSystem
from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect_np


def load_viewmaker_from_checkpoint(viewmaker_cpkt, config_path, eval=True):
    config_path =config_path
    with open(config_path, 'r') as f:
        config_json = json.load(f)
    config = DotMap(config_json)

    SystemClass = globals()[config.system]
    system = SystemClass(config)
    checkpoint = torch.load(viewmaker_cpkt, map_location="cuda:0")
    system.load_state_dict(checkpoint['state_dict'], strict=False)
    if eval:
        viewmaker = system.viewmaker.eval()
    return viewmaker


class CustomDataGenerator(ImageDataGenerator):
    def __init__(self,
                 viewmaker_cpkt=None,
                 config_path=None,
                 **kwargs):
        '''
        Custom image data generator.
        Behaves like ImageDataGenerator, but allows color augmentation.
        '''
        super().__init__(**kwargs)
        self.viewmaker = None
        if viewmaker_cpkt is not None:
            self.viewmaker = load_viewmaker_from_checkpoint(viewmaker_cpkt, config_path, eval=True)

    def flow(self,
             x,
             y=None,
             batch_size=32,
             shuffle=True,
             sample_weight=None,
             seed=None,
             save_to_dir=None,
             save_prefix='',
             save_format='png',
             subset=None):
        data_gen = super().flow(x, y=y, batch_size=batch_size, shuffle=shuffle,
                                sample_weight=sample_weight, seed=seed, save_to_dir=save_to_dir,
                                save_prefix=save_prefix, save_format=save_format, subset=subset)

        def custom_flow():
            for i, (x, y) in enumerate(data_gen):
                torch_tensor = torch.from_numpy(x).permute(0, 3, 1, 2)
                with torch.no_grad():
                    if self.viewmaker is not None:
                        torch_view = self.viewmaker(torch_tensor)
                    else:
                        torch_view = torch_tensor
                tf_view = torch_view.permute(0, 2, 3, 1).numpy()
                if i==0 and not args.debug:
                    grid = np.concatenate((x, tf_view, heatmap_of_view_effect_np(x,tf_view)), 1)[:10]
                    grid = np.moveaxis(grid,0,1)
                    grid = grid.reshape(grid.shape[0], -1, 3)
                    wandb.log({"augmentation sample": wandb.Image(grid)})
                yield (tf_view, y)

        return custom_flow()

class ViewmakerLayer(tf.keras.layers.Layer):
  def __init__(self, viewmaker_cpkt=None,config_path=None,):
    super(ViewmakerLayer, self).__init__()
    self.viewmaker = None
    if viewmaker_cpkt is not None:
        self.viewmaker = load_viewmaker_from_checkpoint(viewmaker_cpkt, config_path, eval=True)

  def call(self, inputs):
    torch_tensor = torch.from_numpy(inputs).permute(0, 3, 1, 2)
    with torch.no_grad():
        if self.viewmaker is not None:
            torch_view = self.viewmaker(torch_tensor)
        else:
            torch_view = torch_tensor
    tf_view = torch_view.permute(0, 2, 3, 1).numpy()
    return tf_view