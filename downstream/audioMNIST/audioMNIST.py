import os

from tensorflow.python.ops.gen_array_ops import Size
os.environ['CUDA_VISIBLE_DEVICES'] = '7'
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = 'true'

import pylab
from matplotlib import pyplot as plt
import numpy as np
import IPython.display as ipd

import scipy.io.wavfile as wav
import wave
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import matplotlib.pyplot as plt
from tqdm import tqdm


data_dir = '/disk2/ofirb/viewmaker/downstream/audioMNIST/free-spoken-digit-dataset-master/recordings'
output_dir =  '/disk2/ofirb/viewmaker/downstream/audioMNIST/output/audio-images'

VER = "VM"
SIZE = 64
BATCH_SIZE = 32
channels = 3
kernel = 4
stride = 1
pool = 2

# process raw data into images
# for filename in tqdm(os.listdir(data_dir)):
#     if "wav" in filename:
#         file_path = os.path.join(data_dir, filename)
#         target_dir = f'class_{filename[0]}'             
#         dist_dir = os.path.join(output_dir, target_dir)
#         file_dist_path = os.path.join(dist_dir, filename)
#         if not os.path.exists(file_dist_path + '.png'):
#             if not os.path.exists(dist_dir):
#                 os.mkdir(dist_dir)                
#             frame_rate, data = wav.read(file_path)
#             signal_wave = wave.open(file_path)
#             sig = np.frombuffer(signal_wave.readframes(frame_rate), dtype=np.int16)
#             fig = plt.figure()
#             plt.specgram(sig, NFFT=1024, Fs=frame_rate, noverlap=900)
#             plt.axis('off')
#             fig.savefig(f'{file_dist_path}.png', dpi=fig.dpi, bbox_inches='tight')
#             plt.close()
# exit()


from viewmaker.downstream.downstream_utils import CustomDataGenerator, ViewmakerLayer

# data_generator = CustomDataGenerator(viewmaker_cpkt="/disk2/ofirb/viewmaker/experiments/experiments/viewmaker-disc-audioMNIST/checkpoints/epoch=199-step=4199.ckpt",
#                                          config_path="/disk2/ofirb/viewmaker/experiments/experiments/viewmaker-disc-audioMNIST/config_vm_audioMNIST.json",
#                                          validation_split=0.2,rescale=1./255)

# data_generator = CustomDataGenerator(validation_split=0.2,rescale=1./255)

# Make a dataset containing the training spectrograms
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
# train_generator = tf.keras.preprocessing.image.DirectoryIterator(
                                            #  image_data_generator=data_generator,
                                             batch_size=BATCH_SIZE,
                                             directory=output_dir,
                                             shuffle=True,
                                             color_mode='rgb',
                                            #  target_size=(SIZE, SIZE),#  image_size=(SIZE, SIZE), validation_split=0.2,
                                             image_size=(SIZE, SIZE), validation_split=0.2,
                                             subset="training",
                                             seed=0)

# Make a dataset containing the validation spectrogram
valid_dataset = tf.keras.preprocessing.image_dataset_from_directory(
# valid_generator = tf.keras.preprocessing.image.DirectoryIterator(
                                            #  image_data_generator=data_generator,
                                             batch_size=BATCH_SIZE,
                                             directory=output_dir,
                                             shuffle=True,
                                             color_mode='rgb',
                                            #  target_size=(SIZE, SIZE),#  image_size=(SIZE, SIZE), validation_split=0.2,
                                             image_size=(SIZE, SIZE), validation_split=0.2,
                                             subset="validation",
                                             seed=0)



# class_names = valid_dataset.class_names
# num_classes = len(class_names)
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_dataset.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = valid_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# model from scratch
model = Sequential([
    layers.experimental.preprocessing.Rescaling(1./1., input_shape=(SIZE, SIZE, channels)),
    # ViewmakerLayer(viewmaker_cpkt="/disk2/ofirb/viewmaker/experiments/experiments/viewmaker-disc-audioMNIST/checkpoints/epoch=199-step=4199.ckpt",
    # config_path="/disk2/ofirb/viewmaker/experiments/experiments/viewmaker-disc-audioMNIST/config_vm_audioMNIST.json"),
    # layers.experimental.preprocessing.Rescaling(1./1., input_shape=(SIZE, SIZE, channels)),
    layers.Conv2D(16, kernel, stride, activation='relu'),
    layers.MaxPool2D(pool),
    layers.Conv2D(32, kernel, stride, activation='relu'),
    layers.MaxPool2D(pool),
    layers.Conv2D(64, kernel, stride, activation='relu'),
    layers.MaxPool2D(pool),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10)
])
model.summary()


# model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
#     filepath='checkpoint.h5',
#     save_weights_only=True,
#     monitor='val_accuracy',
#     mode='max',
#     save_best_only=True)


# Configure the model for training
model.compile(optimizer='adam', 
                #   loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])



epochs = 15
history = model.fit(
    train_ds, epochs=epochs, 
    # train_generator, epochs=epochs, 
    # callbacks=model_checkpoint_callback, 
    validation_data=val_ds,
    # validation_data=valid_generator,
)

# history = model.fit(data_generator.flow(X_train, y_train, batch_size=args.batch_size),
#                         callbacks=callbacks,
#                         steps_per_epoch=100,
#                         epochs=args.num_epochs,
#                         validation_data=(X_validation, y_validation),
#                         verbose=1)




plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
val_txt = f'max_val {np.max(history.history["val_accuracy"]):.3f}'
plt.figtext(0.8, 0.5, val_txt, wrap=True, horizontalalignment='center', fontsize=8)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f"/disk2/ofirb/viewmaker/downstream/audioMNIST/model_accuracy_{SIZE}_{VER}.png")
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig(f"/disk2/ofirb/viewmaker/downstream/audioMNIST/model_loss_{SIZE}_{VER}.png")
plt.clf()