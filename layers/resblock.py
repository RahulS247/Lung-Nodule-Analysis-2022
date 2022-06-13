import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

def ResBlock(input, filter, ratio=16, kernel=3, reg='glorot_uniform', init='zeros'):
    newse_shape = (filter, 1, 1)

    #ResBlock
    x = layers.Conv2D(filter, kernel_size=kernel, kernel_initializer=init, kernel_regularizer=reg, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.Conv2D(filter, kernel_size=kernel, kernel_initializer=init, kernel_regularizer=reg, padding='same')(x)
    x = layers.BatchNormalization()(x)

    if input.shape[0] != filter:
        input = layers.Conv2D(filter, kernel_size=1, padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)

    x = layers.Add()([x, input])
    x = layers.Activation(keras.activations.relu)(x)

    return x