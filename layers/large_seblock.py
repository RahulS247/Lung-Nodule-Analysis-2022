import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

def SeResBlock(input, filter, filter_1,ratio=16, kernel=3, reg='glorot_uniform', init='zeros'):
    newse_shape = (filter, 1, 1)

    #ResBlock
    x = layers.Conv2D(filter_1, kernel_size=1, kernel_initializer=init, kernel_regularizer=reg, padding='same')(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    
    x = layers.Conv2D(filter_1, kernel_size=kernel, kernel_initializer=init, kernel_regularizer=reg, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    
    x = layers.Conv2D(filter, kernel_size=1, kernel_initializer=init, kernel_regularizer=reg, padding='same')(x)
    x = layers.BatchNormalization()(x)

    #SeBlock
    s = layers.GlobalAveragePooling2D()(x)

    s = layers.Dense(filter // ratio, kernel_initializer=init, kernel_regularizer=reg, activation='relu')(s)
    s = layers.Dense(filter, kernel_initializer=init, kernel_regularizer=reg, activation='sigmoid')(s)
    s = layers.Reshape(newse_shape)(s)

    #SeResBlock
    if input.shape[0] != filter:
        input = layers.Conv2D(filter, kernel_size=1, padding='same', kernel_initializer=init, kernel_regularizer=reg)(x)

    x = layers.Multiply()([x, s])
    x = layers.Add()([x, input])
    x = layers.Activation(keras.activations.relu)(x)

    return x