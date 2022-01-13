# Import datasets & libraries
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
from viewmaker.src.systems.image_systems import TransferViewMakerSystem, PretrainViewMakerSystemDisc, \
    PretrainViewMakerSystem
from viewmaker.src.systems.image_systems.utils import heatmap_of_view_effect_np


def define_model():
    model = Sequential()

    model.add(Conv2D(256, (3, 3), padding='same', input_shape=(32, 32, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(512, (3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Flatten())
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization(momentum=0.95,
                                 epsilon=0.005,
                                 beta_initializer=RandomNormal(mean=0.0, stddev=0.05),
                                 gamma_initializer=Constant(value=0.9)))
    model.add(Dense(100, activation='softmax'))
    return model


def load_viewmaker_from_checkpoint(viewmaker_cpkt, config_path, eval=True):
    # base_dir = "/".join(args.ckpt.split("/")[:-2])
    # config_path = os.path.join(base_dir, 'config.json')
    # with open(config_path, 'r') as f:
    #     config_json = json.load(f)
    # config = DotMap(config_json)
    # system = PretrainViewMakerSystem(config)
    # checkpoint = torch.load(viewmaker_cpkt, map_location="cuda:0")
    # system.load_state_dict(checkpoint['state_dict'], strict=False)
    # viewmaker = system.viewmaker.eval()
    # return viewmaker

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


def main():
    if not args.debug:
        from wandb.integration.keras import WandbCallback
        wandb.init(project='transfer_augmentations', name=args.exp_name)

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    if args.low_data:
        len_train, len_test = len(x_train), len(x_test)
        train_idx, test_idx = np.random.choice(range(len_train), len_train//2),    np.random.sample(range(len_test), len_test//2)
        (x_train, y_train), (x_test, y_test) = (x_train[train_idx], y_train[train_idx]), (x_test[test_idx], y_test[test_idx])
    train_images = x_train.astype('float32') / 255
    test_images = x_test.astype('float32') / 255

    # Transform labels to one hot encoding
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)

    model = define_model()
    X_train, X_validation, y_train, y_validation = train_test_split(train_images, train_labels, test_size=0.2,
                                                                    random_state=93)
    if args.expert:
        rotation_range = 20
        horizontal_flip = True
    else:
        rotation_range = 0
        horizontal_flip = False
    data_generator = CustomDataGenerator(viewmaker_cpkt=args.ckpt,
                                         config_path=args.config,
                                         horizontal_flip=horizontal_flip,
                                         rotation_range=rotation_range)
    data_generator.fit(X_train)

    # Configure the model for training
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=args.lr),
                  metrics=['acc',
                           # tfa.metrics.F1Score
                           ])

    callbacks = [WandbCallback()] if not args.debug else []
    history = model.fit(data_generator.flow(X_train, y_train, batch_size=args.batch_size),
                        callbacks=callbacks,
                        steps_per_epoch=100,
                        epochs=args.num_epochs,
                        validation_data=(X_validation, y_validation),
                        verbose=1)

    scores = model.evaluate(test_images, test_labels)
    wandb.log({"validation_acc": scores})


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser()
    arg_parse.add_argument("--exp_name", type=str)
    arg_parse.add_argument("--gpu_device", type=str, default='1')
    arg_parse.add_argument("--ckpt", type=str, default=None)
    arg_parse.add_argument("--config", type=str, default=None)
    arg_parse.add_argument("--debug", action="store_true")
    arg_parse.add_argument("--expert", action="store_true")
    arg_parse.add_argument("--expert", action="store_true")
    arg_parse.add_argument("--num_epochs", type=int, default=350)
    arg_parse.add_argument("--batch_size", type=int, default=64)
    arg_parse.add_argument("--lr", type=float, default=1e-4)
    args = arg_parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

    main()
