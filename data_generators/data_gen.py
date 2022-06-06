import numpy as np
import tensorflow as tf
from tensorflow import keras
from typing import Dict, List, Optional, Tuple, Callable

np.random.seed(42)
tf.random.set_seed(42)


class NoduleDataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(
        self,
        inputs: np.ndarray,
        labels: np.ndarray,
        batch_size: int = 32,
        shuffle: bool = True,
        preprocess_fn: Callable = None,
        seed: np.random.RandomState = None,
    ):
        self._inputs = inputs
        self._labels = labels
        self._preprocess_fn = preprocess_fn
        self._batch_size = batch_size
        self._shuffle = shuffle
        self.indexes = np.arange(len(self._inputs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self._inputs) / self._batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indices = self.indexes[index*self._batch_size:(index+1)*self._batch_size]

        X, y = self._inputs[indices, :], self._labels[indices, :]
        if self._preprocess_fn is not None:
            X = self._preprocess_fn(X)
        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self._inputs))
        if self._shuffle == True:
            np.random.shuffle(self.indexes)