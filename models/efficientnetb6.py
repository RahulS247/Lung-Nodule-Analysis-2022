import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import EfficientNetB6
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

def lung_model(input_shape: int, num_classes: int, verbose: int = 1, keras_aug=False):
    inputs = keras.Input(shape=input_shape)

    # Scaling
    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)

    # Base Model
    base_model = EfficientNetB6(
    include_top=True,
    weights=None,
    input_tensor=None,
    input_shape=input_shape,
    pooling=None,
    classes=num_classes,
    classifier_activation="softmax",
    )
    
    # initializer = tf.keras.initializers.HeNormal()

    # Extended part
    # # x = scale_layer(inputs)
    if keras_aug:
      x = data_augmentaiton(inputs)
      o = base_model(x)
    else:
      o = base_model(inputs)

    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Activation(keras.activations.relu)(x)
    # # x = layers.Flatten()(base_model.output)
    # x = layers.Dense(1024, kernel_initializer=initializer, kernel_regularizer='l2')(x)
    # x = layers.BatchNormalization()(x)
    # x = layers.Activation(keras.activations.relu)(x)
    # x = layers.Dropout(0.2)(x)
    # o = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer, kernel_regularizer='l2')(x)

    # Build model
    model = keras.Model(inputs=inputs, outputs=o)

    print(model.summary())

    return model

def data_augmentaiton(inputs):
  x = layers.RandomFlip(mode='horizontal_and_vertical')(inputs)
  x = layers.RandomRotation(factor=0.5)(x)
  x = layers.GaussianNoise(stddev=0.2)(x)
  return x