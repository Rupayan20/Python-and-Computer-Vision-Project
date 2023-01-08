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
