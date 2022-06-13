import pathlib
from pathlib import Path

from typing import Tuple
from enum import Enum, unique

import numpy as np
import matplotlib.pyplot as plt

from random import sample, choice, random
import cv2 as cv
from skimage.util import random_noise
from skimage.transform import rotate

import tensorflow
import tensorflow.keras
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import optimizers
from tensorflow.keras import losses
from tensorflow.keras.applications import VGG16, ResNet50, EfficientNetB0, ResNet101, ResNet152 
from tensorflow.keras.optimizers import SGD, Adam

from tensorflow.keras.losses import categorical_crossentropy

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

import tensorflow_addons as tfa

from utils import maybe_download_vgg16_pretrained_weights

import os
import itertools
import json

import click

# models
# from models.resnet import lung_model
# from models.simple_model import lung_model
# from models.vgg16 import lung_model
# from models.seresnet18 import lung_model
# from models.resnet_baseOnly import lung_model
# from models.resnet18 import lung_model

# Data generation
from data_generators.data_gen import NoduleDataGenerator
from balanced_sampler import sample_balanced, UndersamplingIterator
from data import load_dataset
from data_generators import data_augmentation as data_aug

import sklearn
from sklearn import metrics as skm

import itertools
import os

data_part_options = ["slices", 'full']


@click.command()
@click.option(
    "--raw_data_dir",
    type=pathlib.Path,
    required=True,
    help="path to root folder containing the data set",
)
@click.option(
    "--gen_data_dir",
    type=pathlib.Path,
    required=True,
    help="path to root folder for the augmented data set",
)
@click.option(
    "--out_dir",
    type=pathlib.Path,
    required=True,
    help="path to root folder for the output files",
)
@click.option(
    "--input_size",
    type=int,
    default=224,
    help="The inputsize of the images for the network",
)
@click.option(
    "--batch_size",
    type=int,
    default=128,
    help="The batch size for the malignancy prediction",
)
@click.option(
    "--base_model",
    type=str,
    default="vgg16",
    help="The pretrained base model of the network",
)
#cross_slices_only#fully
@click.option(
    "--data_part",
    type=click.Choice(data_part_options),
    default="slices",
    help="The pretrained base model of the network",
)
@click.option(
    "--val_fraciton",
    type=float,
    default=0.15,
    help="The train val split fraction",
)
@click.option('--problem',
              type=click.Choice(['malignancy', 'noduletype'], case_sensitive=True),
              required=True,
              help="If this run should consider maligancy or the nodule types",
              )
@click.option(
    "--learning_rate",
    type=float,
    default=0.0001,
)
@click.option(
    "--epochs",
    type=int,
    default=250,
)
@click.option(
    "--early_stop_delta",
    type=float,
    default=0,
)
@click.option(
    "--kfolds",
    type=int,
    default=1,
)
@click.option(
    "--stratified",
    type=bool,
    default=False
)
@click.option(
    "--sample_strat",
    type=click.Choice(['undersampling', 'normal'], case_sensitive=True),
    default='undersampling'
)
@click.option(
    "--run_name",
    type=str,
    default="model_1"
)
@click.option(
    "--preprocessing_type",
    type=click.Choice(['heavy', 'normal', "blur"], case_sensitive=True),
    default="normal"
)
def main(
    raw_data_dir: pathlib.Path,
    gen_data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    input_size: int,
    batch_size: int,
    base_model: str,
    data_part:str,
    val_fraciton: float,
    problem: str,
    learning_rate: float,
    epochs: int,
    early_stop_delta: float,
    kfolds: int,
    stratified: bool,
    sample_strat: str,
    run_name: str,
    preprocessing_type: str,
):
    
    if base_model == "resnet18":
        from models.resnet18 import lung_model
    elif base_model == "resnet50":
        from models.resnet_baseOnly import lung_model
    elif base_model == "vgg16" or base_model == 'vgg16_pretrained':
        from models.vgg16 import lung_model
    elif base_model == "vgg19":
        from models.vgg19 import lung_model
    elif base_model == "inceptionresnet":
        from models.inceptionResNet import lung_model
    elif base_model == "densenet":
        from models.densenet import lung_model
    elif base_model == "simplecnn":
        from models.simple_model import lung_model
    elif base_model == "efficientnetb0":
        from models.efficientnetb0 import lung_model
    elif base_model == "efficientnetb0v2":
        from models.efficientnetb0v2 import lung_model
    elif base_model == "efficientnetb1":
        from models.efficientnetb1 import lung_model
    elif base_model == "efficientnetb1v2":
        from models.efficientnetb1v2 import lung_model
    elif base_model == "efficientnetb2":
        from models.efficientnetb2 import lung_model
    elif base_model == "efficientnetb2v2":
        from models.efficientnetb2v2 import lung_model
    elif base_model == "efficientnetb3":
        from models.efficientnetb3 import lung_model
    elif base_model == "efficientnetb3v2":
        from models.efficientnetb3v2 import lung_model
    elif base_model == "efficientnetb4":
        from models.efficientnetb4 import lung_model
    elif base_model == "efficientnetb5":
        from models.efficientnetb5 import lung_model
    elif base_model == "efficientnetb6":
        from models.efficientnetb6 import lung_model
    elif base_model == "efficientnetb7":
        from models.efficientnetb7 import lung_model
    elif base_model == "seresnet50":
        from models.seresnet18 import lung_model
    elif base_model == "seresnet18":
        from models.seresnet50 import lung_model

    tensorflow.random.set_seed(42)
    np.random.seed(42)
    # Enforce some Keras backend settings that we need
    tensorflow.keras.backend.set_image_data_format("channels_first")
    tensorflow.keras.backend.set_floatx("float32")
    print("num GPU's Available:", len(tensorflow.config.list_physical_devices('GPU')))
    print(tensorflow.config.list_physical_devices('GPU'))

    # This should point at the directory containing the source LUNA22 prequel dataset
    DATA_DIRECTORY = raw_data_dir#Path().absolute() / "LUNA22 prequel"

    # This should point at a directory to put the preprocessed/generated datasets from the source data
    GENERATED_DATA_DIRECTORY = gen_data_dir#Path().absolute()

    # This should point at a directory to store the training output files
    TRAINING_OUTPUT_DIRECTORY = out_dir#Path().absolute()

    # This should point at the pretrained model weights file for the VGG16 model.
    # The file can be downloaded here:
    # https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    if base_model == 'vgg16_pretrained':
        PRETRAINED_VGG16_WEIGHTS_FILE = (
            Path().absolute()
            / "pretrained_weights"
            / "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
        )
        maybe_download_vgg16_pretrained_weights(PRETRAINED_VGG16_WEIGHTS_FILE)
    if base_model == "efficientnetb7":
        PRETRAINED_efficientb7_WEIGHTS_FILE = (
          Path().absolute()
          / "pretrained_weights"
          / "efficientB0_weight.h5"
        )

    # Load dataset
    # This method will generate a preprocessed dataset from the source data if it is not present (only needs to be done once)
    # Otherwise it will quickly load the generated dataset from disk
    cross_slices_only = (data_part=="slices")
    full_dataset = load_dataset(
        input_size=input_size,#224,
        new_spacing_mm=0.2,
        cross_slices_only=cross_slices_only,
        generate_if_not_present=True,
        always_generate=False,
        source_data_dir=DATA_DIRECTORY,
        generated_data_dir=GENERATED_DATA_DIRECTORY, #Why isnt it loaded???
    )
    inputs = full_dataset["inputs"]
    
    input_shape = 3, input_size, input_size

    @unique
    class MLProblem(Enum):
        malignancy_prediction = "malignancy"
        nodule_type_prediction = "noduletype"


    # Here you can switch the machine learning problem to solve
    if problem == "noduletype":
        problem = MLProblem.nodule_type_prediction
    else:
        problem = "malignancy"#MLProblem.malignancy_prediction

    # Configure problem specific parameters
    if problem =="malignancy": #"malignancy"MLProblem.malignancy_prediction:
        # We made this problem a binary classification problem:
        # 0 - benign, 1 - malignant
        num_classes = 2
        batch_size = batch_size#32
        # Take approx. 15% of all samples for the validation set and ensure it is a multiple of the batch size
        num_validation_samples = int(len(inputs) * val_fraciton / batch_size) * batch_size
        labels = full_dataset["labels_malignancy"]
        # It is possible to generate training labels yourself using the raw annotations of the radiologists...
        labels_raw = full_dataset["labels_malignancy_raw"]
        classes_names = ["Benign", "Malignant"]
    elif problem == MLProblem.nodule_type_prediction:
        # We made this problem a multiclass classification problem with three classes:
        # 0 - non-solid, 1 - part-solid, 2 - solid
        num_classes = 3
        batch_size = batch_size #30  # make this a factor of three to fit three classes evenly per batch during training
        # This dataset has only few part-solid nodules in the dataset, so we make a tiny validation set
        num_validation_samples = batch_size * 2
        labels = full_dataset["labels_nodule_type"]
        # It is possible to generate training labels yourself using the raw annotations of the radiologists...
        labels_raw = full_dataset["labels_nodule_type_raw"]
        classes_names = ["Nonsolid", "PartSolid", "Solid"]

    else:
        raise NotImplementedError(f"An unknown MLProblem was specified: {problem}")

    print(
        f"Finished loading data for MLProblem: {problem}... X:{inputs.shape} Y:{labels.shape}"
    )

    # partition small and class balanced validation set from all data
    validation_indices = sample_balanced(
        input_labels=np.argmax(labels, axis=1),
        required_samples=num_validation_samples,
        class_balance=None,  # By default sample with equal probability, e.g. for two classes : {0: 0.5, 1: 0.5}
        shuffle=True,
    )
    validation_mask = np.isin(np.arange(len(labels)), list(validation_indices.values()))
    training_inputs = inputs[~validation_mask, :]
    training_labels = labels[~validation_mask, :]
    validation_inputs = inputs[validation_mask, :]
    validation_labels = labels[validation_mask, :]

    print(f"Splitted data into training and validation sets:")
    training_class_counts = np.unique(
        np.argmax(training_labels, axis=1), return_counts=True
    )[1]
    validation_class_counts = np.unique(
        np.argmax(validation_labels, axis=1), return_counts=True
    )[1]
    print(
        f"Training   set: {training_inputs.shape} {training_labels.shape} {training_class_counts}"
    )
    print(
        f"Validation set: {validation_inputs.shape} {validation_labels.shape} {validation_class_counts}"
    )


    # Split dataset into two data generators for training and validation
    # Technically we could directly pass the data into the fit function of the model
    # But using generators allows for a simple way to add augmentations and preprocessing
    # It also allows us to balance the batches per class using undersampling

    # The following methods can be used to implement custom preprocessing/augmentation during training
    if preprocessing_type == "normal":
        train_preprocess_fn = data_aug.normal_train_preprocess_fn
        validation_preprocess_fn = data_aug.validation_preprocess_fn
    elif preprocessing_type == "blur":
        train_preprocess_fn = data_aug.blurr_train_preprocess_fn
        validation_preprocess_fn = data_aug.validation_preprocess_fn
    elif preprocessing_type == "heavy":
        train_preprocess_fn = data_aug.heavy_train_preprocess_fn
        validation_preprocess_fn = data_aug.validation_preprocess_fn

    if sample_strat == "undersampling":
        training_data_generator = UndersamplingIterator(
            training_inputs,
            training_labels,
            shuffle=True,
            preprocess_fn=train_preprocess_fn,
            batch_size=batch_size,
        )
        validation_data_generator = UndersamplingIterator(
            validation_inputs,
            validation_labels,
            shuffle=False,
            preprocess_fn=validation_preprocess_fn,
            batch_size=batch_size,
        )
    elif sample_strat == "normal":
        training_data_generator = NoduleDataGenerator(
            inputs=training_inputs,
            labels=training_labels,
            batch_size=batch_size,
            shuffle=True,
            preprocess_fn=train_preprocess_fn,
        )
        
        validation_data_generator = NoduleDataGenerator(
            inputs=validation_inputs,
            labels=validation_labels,
            batch_size=batch_size,
            shuffle=True,
            preprocess_fn=validation_preprocess_fn,
        )

    # Load Model 
    model = lung_model(input_shape, num_classes)
    fold_var = 1

    # Load the pretrained imagenet VGG model weights except for the last layer
    # Because the pretrained weights will have a data size mismatch in the last layer of our model
    # two warnings will be raised, but these can be safely ignored.
    if base_model == 'vgg16_pretrained':
        model.load_weights(str(PRETRAINED_VGG16_WEIGHTS_FILE), by_name=True, skip_mismatch=True)
    if base_model == 'efficientnetb7_pretrained':
        model.load_weights(str(PRETRAINED_efficientb7_WEIGHTS_FILE), by_name=True, skip_mismatch=True)

    # Prepare model for training by defining the loss, optimizer, and metrics to use
    # Output labels and predictions are one-hot encoded, so we use the categorical_accuracy metric
    opt = optimizers.Adam(learning_rate=learning_rate)
    opt = tfa.optimizers.Lookahead(
        optimizer=opt,
        sync_period=5,
        slow_step_size=0.5
    )
    
    if problem == MLProblem.malignancy_prediction:
        loss_fn = losses.categorical_crossentropy
        class_weight = {0:1.5, 1:0.7}
    else:
        loss_fn = losses.categorical_crossentropy
        class_weight = {0:4.96, 1:10.3, 2:0.37}
    
    if sample_strat == "undersampling":
        class_weight = None
    
    model_metrics = [
      keras.metrics.TruePositives(name='tp'),
      keras.metrics.FalsePositives(name='fp'),
      keras.metrics.TrueNegatives(name='tn'),
      keras.metrics.FalseNegatives(name='fn'), 
      keras.metrics.CategoricalAccuracy(name='categorical_accuracy'),
      keras.metrics.Precision(name='precision'),
      keras.metrics.Recall(name='recall'),
      keras.metrics.AUC(name='auc'),
      keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
    ]
    
    model.compile(
        optimizer=opt,#optimizers.SGD(learning_rate=learning_rate, momentum=0.8, nesterov=True),
        loss=loss_fn,
        metrics=model_metrics,#["categorical_accuracy"],
    )
    
    summary = model.summary()
    print(summary)




    # Start actual training process
    output_model_file = (
       TRAINING_OUTPUT_DIRECTORY / base_model / get_model_name(fold_var, problem, run_name) #f"vgg16_{problem.value}_best_val_accuracy.h5"
    )
    
    p = pathlib.Path(TRAINING_OUTPUT_DIRECTORY / f"logs/{problem.value}/{base_model}")
    p.mkdir(parents=True, exist_ok=True)
    omf = pathlib.Path(TRAINING_OUTPUT_DIRECTORY / base_model)
    omf.mkdir(parents=True, exist_ok=True)
    
    n_dirs = len(os.listdir(TRAINING_OUTPUT_DIRECTORY / f"logs/{problem.value}/{base_model}"))
    logdir = TRAINING_OUTPUT_DIRECTORY / f"logs/{problem.value}/{base_model}/_{run_name}__{str(fold_var)}_run{n_dirs}"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)
    
    callbacks = [
        TerminateOnNaN(),
        ModelCheckpoint(
            str(output_model_file),
            monitor="val_categorical_accuracy",
            mode="auto",
            verbose=1,
            save_best_only=True,
            save_weights_only=False,
            save_freq="epoch",
        ),
        EarlyStopping(
            monitor="val_categorical_accuracy",
            mode="auto",
            min_delta=early_stop_delta,
            patience=100,
            verbose=1,
        ),
        tensorboard_callback
    ]
    
    history = model.fit(
        training_data_generator,
        # x=training_inputs,
        # y=training_labels,
        # batch_size=batch_size,
        steps_per_epoch=len(training_data_generator),
        class_weight=class_weight,
    ]
    history = model.fit(
        training_data_generator,
        steps_per_epoch=len(training_data_generator),
        validation_data=validation_data_generator,
        validation_steps=None,
        validation_freq=1,
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
        shuffle=True,
    )
    
    #Load best model
    model.load_weights(output_model_file)
    
    # Conf matrix
    y_val_auto = model.predict(validation_data_generator)
    print(validation_labels.shape, y_val_auto.shape)
    print(validation_labels.argmax(axis=1).shape, y_val_auto.argmax(axis=1).shape)
    print("IMAGE")
    print(np.max(training_data_generator.__getitem__(0)[0]))
    conf_mat_nn = skm.confusion_matrix(validation_labels.argmax(axis=1), y_val_auto.argmax(axis=1))
    acc_nn = skm.accuracy_score(validation_labels.argmax(axis=1), y_val_auto.argmax(axis=1))
    print("vall acc:", acc_nn)
    output_conf_img_file = (
        TRAINING_OUTPUT_DIRECTORY / base_model / f"{run_name}_{problem.value}_conf_plot.png"
    )
    plot_confusion_matrix(conf_mat_nn, classes_names, output_conf_img_file)


    # generate a plot using the training history...
    output_history_img_file = (
        TRAINING_OUTPUT_DIRECTORY / base_model / f"{run_name}_{problem.value}_train_plot.png"
    )
    print(f"Saving training plot to: {output_history_img_file}")
    plt.plot(history.history["categorical_accuracy"])
    plt.plot(history.history["val_categorical_accuracy"])
    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.title("model accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("Epoch")
    plt.legend(["Accuracy", "Validation Accuracy", "loss", "Validation Loss"])
    plt.savefig(str(output_history_img_file), bbox_inches="tight")

def get_model_name(fold_var, problem, run_name):
    return f"{run_name}_{problem.value}_{str(fold_var)}_best_vall_acc.h5"

def plot_confusion_matrix(conf_mat, classes, outfile,title='Confusion Matrix', cmap=plt.cm.Blues,):
    """
    This function prints and plots the confusion matrix
    """
    plt.imshow(conf_mat, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)

    thresh = conf_mat.max() / 2.
    for i, j in itertools.product(range(conf_mat.shape[0]), range(conf_mat.shape[1])):
        plt.text(j, i, conf_mat[i, j], horizontalalignment="center",
                 color="white" if conf_mat[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(str(outfile), bbox_inches="tight")

if __name__ == "__main__":
    main()