# Import datasets & libraries
import argparse
import os

import wandb

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
import time
from tensorflow.keras import optimizers
import tensorflow_addons as tfa

# Normalize images

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

def main():
    if not args.debug:
        from wandb.integration.keras import WandbCallback
        wandb.init(project='transfer_augmentations', name=args.exp_name)

    (x_train, y_train), (x_test, y_test) = cifar100.load_data()
    train_images = x_train.astype('float32') / 255
    test_images = x_test.astype('float32') / 255

    # Transform labels to one hot encoding
    train_labels = to_categorical(y_train)
    test_labels = to_categorical(y_test)

    model = define_model()
    X_train, X_validation, y_train, y_validation = train_test_split(train_images, train_labels, test_size=0.2,
                                                                    random_state=93)
    data_generator = ImageDataGenerator(rotation_range=20, horizontal_flip=True)
    data_generator.fit(X_train)

    # Configure the model for training
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.RMSprop(learning_rate=args.lr),
                  metrics=['acc', tfa.metrics.F1Score])

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
    arg_parse.add_argument("--debug", action="store_true")
    arg_parse.add_argument("--num_epochs", type=int, default=350)
    arg_parse.add_argument("--batch_size", type=int, default=64)
    arg_parse.add_argument("--lr", type=float, default=1e-4)
    arg_parse.add_argument("--gpu_device", type=str, default='0')
    args = arg_parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_device

    main()
