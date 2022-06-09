import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

def lung_model(input_shape: int, num_classes: int, verbose: int = 1, keras_aug=False):
    inputs = keras.Input(shape=input_shape)

    # Scaling
    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)
    initializer = tf.keras.initializers.HeNormal()

    # Extended part
    # x = scale_layer(inputs)
    if keras_aug:
      x = data_augmentaiton(inputs)
      x = layers.Conv2D(16, 3, padding="valid", kernel_initializer=initializer, kernel_regularizer='l2')(x)
    else:
      x = layers.Conv2D(16, 3, padding="valid", kernel_initializer=initializer, kernel_regularizer='l2')(inputs)
      
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(32, 3, padding="valid", kernel_initializer=initializer, kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.AveragePooling2D()(x)
    x = layers.Conv2D(64, 3, padding="valid", kernel_initializer=initializer, kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    
    x = layers.Dense(128, kernel_initializer=initializer, kernel_regularizer='l2')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation(keras.activations.relu)(x)
    x = layers.Dropout(0.5)(x)
    o = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer, kernel_regularizer='l2')(x)
    # x = layers.Flatten()(base_model.output)

    # Build model
    model = keras.Model(inputs=inputs, outputs=o)

    print(model.summary())

    return model

def data_augmentaiton(inputs):
  x = layers.RandomFlip(mode='horizontal_and_vertical')(inputs)
  x = layers.RandomRotation(factor=0.5)(x)
  x = layers.GaussianNoise(stddev=0.2)(x)
  return x