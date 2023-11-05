import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import datetime
from tensorboard import program

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pathlib

#dataset_url= "https://storage.googleapis.com/download.\
#tensorflow.org/example_images/flower_photos.tgz"

#data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
#data_dir = pathlib.Path(data_dir)

#image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

#roses = list(data_dir.glob('roses/*'))
#PIL.Image.open(str(roses[0]))

#Training split
#train_ds = tf.keras.utils.image_dataset_from_directory(
    #data_dir,
    #validation_split=0.2,
    #seed = 123,
    #image_size=(180, 180),
    #batch_size=32)

#Validation Split
#val_ds = tf.keras.utils.image_dataset_from_directory(
    #data_dir,
    #validation_split=0.2,
    #subset="validation",
    #seed=123,
    #image_size=(180, 180),
    #batch_size=32
#)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0


def create_model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28), name='layers_flatten'),
        tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
        tf.keras.layers.Dropout(0.2, name='layers_dropout'),
        tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
    ])

model = create_model()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%d%m%Y-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

model.fit(x = x_train,
          y = y_train,
          epochs=5,
          validation_data=(x_test, y_test),
          callbacks=[tensorboard_callback])

tracking_address = log_path

if __name__ == "__maain__":
    tb = program.Tensorboard()
    tb.configure(argv=[None, '--logdir', tracking_address])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")