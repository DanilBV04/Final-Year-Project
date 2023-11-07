import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
keras = tf.keras
K = keras.backend
KL = keras.layers
Lambda, Input = KL.Lambda
Flatten = KL.Flatten
Lambda = KL.Lambda
Input = KL.Input
Model = keras.model
import datetime
from tensorboard import program
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Input, Conv2D, MaxPooling2D, Dropout
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras import regularizers
import pathlib

# dataset_url= "https://storage.googleapis.com/download.\
# tensorflow.org/example_images/flower_photos.tgz"

# data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
# data_dir = pathlib.Path(data_dir)

# image_count = len(list(data_dir.glob('*/*.jpg')))
# print(image_count)

# roses = list(data_dir.glob('roses/*'))
# PIL.Image.open(str(roses[0]))

# Training split
# train_ds = tf.keras.utils.image_dataset_from_directory(
# data_dir,
# validation_split=0.2,
# seed = 123,
# image_size=(180, 180),
# batch_size=32)

# Validation Split
# val_ds = tf.keras.utils.image_dataset_from_directory(
# data_dir,
# validation_split=0.2,
# subset="validation",
# seed=123,
# image_size=(180, 180),
# batch_size=32
# )

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=1)
x_train, x_test = x_train / 255.0, x_test / 255.0


def model():
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(32, 32, 3), name='layers_flatten'),
        tf.keras.layers.Dense(128, activation='relu', name='layers_dense'),
        tf.keras.layers.Dropout(0.2, name='layers_dropout'),
        tf.keras.layers.Dense(256, activation='relu', name='layers_dense'),
        tf.keras.layers.Dropout(0.2, name='layers_dropout'),
        tf.keras.layers.Dense(512, activation='relu', name='layers_dense'),
        tf.keras.layers.Dropout(0.2, name='layers_dropout'),
        tf.keras.layers.Dense(10, activation='softmax', name='layers_dense_2')
    ])


model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
y_val = tf.keras.utils.to_categorical(y_val,num_classes=10)

a = model().fit(x_train, y_train, epochs=12, validation_data=(x_val, y_val), batch_size=16, verbose=1)


def display_results(model_history):
    plt.plot(model_history.history['accuracy'], label='Traiining Accuracy', c='blue', ls='-')
    plt.plot(model_history.history['loss'], label='Training Loss', c='blue', ls='--')
    plt.plot(model_history.history['val_accuracy'], label='Validation Accuracy', c='orange', ls='-')
    plt.plot(model_history.history['val_loss'], label='Validation Loss', c='orange', ls='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()


display_results(a)