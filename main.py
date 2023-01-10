#import libraries
import tensorflow as tf
from tensorflow import keras
import numpy as np

fashion_mnist= keras.datasets.fashion_mnist
(train_images,train_labels),(test_images,test_labels)= fashion_mnist.load_data()
train_images= train_images/255.0
train_labels= train_images/255.0
train_images[0].shape

train_images= train_images.reshape(len(train_images),28,28,1)
test_images= test_images.reshape(len(test_images),28,28,1)


def build_model(hp):
  model= keras.Sequential([
      keras.layers.Conv2D(
          filters= hp.Int('conv_1_filter', min_value=32, max_value=128, step=16),
          kernel_size= hp.Choice('conv_1_kernel', values=[3,5]),
          activation='relu',
          input_shape= (28,28,1)
      ),
      keras.layers.Conv2D(
          filters= hp.Int('conv_2_filter', min_value=32, max_value=128, step=16),
          kernel_size= hp.Choice('conv_2_kernel', values=[3,5]),
          activation='relu'
      ),
      keras.layers.Flatten(),
      keras.layers.Dense(
        units= hp.Int('dense_1_units', min_value=32, max_value=128, step=16),
        activation='relu'
      ),
      keras.layers.Dense(10, activation='softmax')  #output layer
  ])

  model.compile(optimizer= keras.optimizers.Adam(hp.Choice('learning rate', values=[1e-2, 1e-3])),
              loss= 'sparse_categorical_crossentropy',
              metrics=['accurecy'])
  
  return model

from kerastuner import RandomSearch
from kerastuner.engine.hyperparameters import HyperParameters
