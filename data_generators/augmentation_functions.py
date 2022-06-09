from random import sample, choice, random
import cv2 as cv
from skimage.util import random_noise
from skimage import transform
from skimage import exposure
import numpy as np

from typing import Tuple

def clip_and_scale(
    data: np.ndarray, min_value: float = -1000.0, max_value: float = 400.0
) -> np.ndarray:
    data = (data - min_value) / (max_value - min_value)
    data[data > 1] = 1.0
    data[data < 0] = 0.0
    return data

def rotation_augmentation(input_sample: np.ndarray    
                            ) -> np.ndarray:
    angles = [90,180,270]
    angle = sample(angles,1)[0]
    input_sample = transform.rotate(input_sample, angle)        
    return input_sample           

def flip_augmentation(input_sample: np.ndarray
                        ) -> np.ndarray:
    axis=(1,2)
    if np.random.random_sample() > 0.5:
        input_sample = np.flip(input_sample, axis=axis[0])
    else:
        input_sample = np.flip(input_sample, axis=axis[1])
    return input_sample

def add_noise_augmentation(input_sample: np.ndarray    
                            ) -> np.ndarray:
    input_sample = random_noise(input_sample)
    return input_sample 

def blur_augmentation(input_sample: np.ndarray    
                            ) -> np.ndarray:
        input_sample = cv.GaussianBlur(input_sample, (9,9),0)
        return input_sample

def random_flip_augmentation(
    input_sample: np.ndarray, axis: Tuple[int, ...] = (1, 2)
) -> np.ndarray:
    for ax in axis:
        if np.random.random_sample() > 0.5:
            input_sample = np.flip(input_sample, axis=ax)
    return input_sample

def img_exposure_gamma(image,gamma):
    img = exposure.adjust_gamma(image,gamma)
    return img

def img_exposure_log(image, log):
    img = exposure.adjust_log(image,log)
    return img

def img_exposure_sigmoid(image, sigmoid):
    img = exposure.adjust_sigmoid(image,sigmoid)
    return img