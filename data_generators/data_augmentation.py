from random import sample, choice, random
import cv2 as cv
from skimage.util import random_noise
from skimage import transform
from skimage import exposure
import numpy as np

from typing import Tuple
from data_generators import augmentation_functions as data_aug_f


def shared_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    """Preprocessing that is used by both the training and validation sets during training

    :param input_batch: np.ndarray [batch_size x channels x dim_x x dim_y]
    :return: np.ndarray preprocessed batch
    """
    input_batch = data_aug_f.clip_and_scale(input_batch, min_value=-1000.0, max_value=400.0)
    # Can add more preprocessing here...
    return input_batch


def normal_train_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)

    output_batch = []
    for sample in input_batch:
        sample = data_aug_f.rotation_augmentation(sample) if random()<0.25 else sample
        sample = data_aug_f.flip_augmentation(sample) if random()<0.75 else sample
        sample = data_aug_f.add_noise_augmentation(sample) if random()<0.5 else sample
        # sample = data_aug.blur_augmentation(sample) if random()<0.25 else sample
        # sample = data_aug.img_exposure_gamma(sample, 0.8) if random()<0.125 else sample
        # sample = data_aug.img_exposure_gamma(sample, 1.2) if random()<0.125 else sample
        # sample = data_aug.img_exposure_log(sample, 0.8) if random()<0.125 else sample
        # sample = data_aug.img_exposure_log(sample, 1.2) if random()<0.125 else sample
        output_batch.append(sample)

    return np.array(output_batch)

def blurr_train_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)

    output_batch = []
    for sample in input_batch:
        sample = data_aug_f.rotation_augmentation(sample) if random()<0.25 else sample
        sample = data_aug_f.flip_augmentation(sample) if random()<0.75 else sample
        sample = data_aug_f.add_noise_augmentation(sample) if random()<0.5 else sample
        sample = data_aug_f.blur_augmentation(sample) if random()<0.25 else sample
        # sample = data_aug.img_exposure_gamma(sample, 0.8) if random()<0.125 else sample
        # sample = data_aug.img_exposure_gamma(sample, 1.2) if random()<0.125 else sample
        # sample = data_aug.img_exposure_log(sample, 0.8) if random()<0.125 else sample
        # sample = data_aug.img_exposure_log(sample, 1.2) if random()<0.125 else sample
        output_batch.append(sample)

    return np.array(output_batch)

def heavy_train_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)

    output_batch = []
    for sample in input_batch:
        sample = data_aug_f.rotation_augmentation(sample) if random()<0.25 else sample
        sample = data_aug_f.flip_augmentation(sample) if random()<0.75 else sample
        sample = data_aug_f.add_noise_augmentation(sample) if random()<0.5 else sample
        sample = data_aug_f.blur_augmentation(sample) if random()<0.25 else sample
        sample = data_aug_f.img_exposure_gamma(sample, 0.8) if random()<0.125 else sample
        sample = data_aug_f.img_exposure_gamma(sample, 1.2) if random()<0.125 else sample
        sample = data_aug_f.img_exposure_log(sample, 0.8) if random()<0.125 else sample
        sample = data_aug_f.img_exposure_log(sample, 1.2) if random()<0.125 else sample
        output_batch.append(sample)

    return np.array(output_batch)


def validation_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
    input_batch = shared_preprocess_fn(input_batch=input_batch)
    return input_batch