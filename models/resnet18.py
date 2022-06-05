import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

from layers.resblock import ResBlock

def lung_model(data_size_in, n_classes: int, verbose: int = 1) -> keras.Model:
      
    inputs = layers.Input(shape=data_size_in)
    
    init = tf.keras.initializers.HeNormal()
    reg = tf.keras.regularizers.l2(0.001)
    
    #x = layers.RandomFlip(mode='horizontal')(inputs)
    x = data_augmentaiton(inputs)
    
    x = layers.Conv2D(64, kernel_size=7, strides=2, kernel_initializer=init, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.MaxPooling2D(pool_size=3, strides=2)(x)
    
    x = ResBlock(x, 64, reg=reg, init=init)
    x = ResBlock(x, 64, reg=reg, init=init)

    x = ResBlock(x, 128, reg=reg, init=init)
    x = ResBlock(x, 128, reg=reg, init=init)

    x = ResBlock(x, 256, reg=reg, init=init)
    x = ResBlock(x, 256, reg=reg, init=init)

    x = ResBlock(x, 512, reg=reg, init=init)
    x = ResBlock(x, 512, reg=reg, init=init)
    
    x = layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Flatten()(x)

    x = layers.Dense(1024, kernel_initializer=init, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.Dropout(0.5)(x)
    
    o = layers.Dense(n_classes, activation='softmax', kernel_regularizer=reg)(x)
    
    model = keras.Model(inputs=inputs, outputs=o)
    
    return model

def data_augmentaiton(inputs):
  x = layers.RandomFlip(mode='horizontal_and_vertical')(inputs)
  x = layers.RandomRotation(factor=0.5)(x)
  x = layers.GaussianNoise(stddev=0.2)(x)
  return x