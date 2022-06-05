import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16, ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN



def lung_model(input_shape: int, num_classes: int, verbose: int = 1):
    inputs = keras.Input(shape=input_shape)

    # Scaling
    # scale_layer = keras.layers.Rescaling(scale=1 / 127.5, offset=-1)

    # Base Model
    base_model = ResNet50(
       include_top=False,
       weights="imagenet",
       input_tensor=None,
       input_shape=input_shape,
       pooling=None ,
       classes=num_classes,
       classifier_activation="softmax",
    )
    
    initializer = tf.keras.initializers.HeNormal()

    # Extended part
    # x = scale_layer(inputs)
    x = data_augmentaiton(inputs)
    x = base_model(x)
    x = layers.Flatten()(x)
    x = layers.Dropout(0.2)(x)
    o = layers.Dense(num_classes, activation='softmax', kernel_initializer=initializer, kernel_regularizer='l2')(x)

    # Build model
    model = keras.Model(inputs=inputs, outputs=o)

    print(model.summary())

    return model

def data_augmentaiton(inputs):
  x = layers.RandomFlip(mode='horizontal_and_vertical')(inputs)
  x = layers.RandomRotation(factor=0.5)(x)
  x = layers.GaussianNoise(stddev=0.2)(x)
  return x