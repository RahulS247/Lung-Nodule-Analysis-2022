import pathlib
from typing import Tuple
from enum import Enum, unique

import numpy as np
import matplotlib.pyplot as plt


import tensorflow
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.applications import VGG16, ResNet50, ResNet101, ResNet152 
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, TerminateOnNaN

from balanced_sampler import sample_balanced, UndersamplingIterator
from data import load_dataset
from utils import maybe_download_vgg16_pretrained_weights

import os
import itertools
import json

import click


data_part_options = ["slices", 'full']
base_model_options = ["vgg16","ResNet152","ResNet101","ResNet50","ffncn"]
problem_options = ['malignancy', 'noduletype']

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
    "--existing_dir",
    type=pathlib.Path,
    default="",
    help="check if this training combination in path allready exist in this path",
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
    "--base_model_name",
    type=click.Choice(base_model_options),
    default=base_model_options[0],
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
              type=click.Choice(problem_options, case_sensitive=True),
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
    "--early_stop_delta",
    type=float,
    default=0,
)
@click.option('--all_combinations',  is_flag=True, help="Print more output.")





def main(
    raw_data_dir: pathlib.Path,
    gen_data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    existing_dir: pathlib.Path,
    input_size: int,
    batch_size: int,
    base_model_name: str,
    data_part:str,
    val_fraciton: float,
    problem: str,
    learning_rate: float,
    epochs: int,
    early_stop_delta: float,
    all_combinations:bool = False,
    
):
    print("all combos: ", all_combinations)
    if(all_combinations):
        for i in itertools.product(base_model_options, data_part_options,[problem_options[0]]):
            print(i)
            #main(raw_data_dir,gen_data_dir=gen_data_dir,out_dir=out_dir,existing_dir=existing_dir,input_size=input_size,batch_size=batch_size,base_model_name=i[0],data_part=i[1],val_fraciton=val_fraciton,problem=i[2],learning_rate=learning_rate,epochs=epochs,early_stop_delta=early_stop_delta, all_combinations=False)
            try:
                train_and_eval_network(raw_data_dir,gen_data_dir=gen_data_dir,out_dir=out_dir,existing_dir=existing_dir,input_size=input_size,batch_size=batch_size,base_model_name=i[0],data_part=i[1],val_fraciton=val_fraciton,problem=i[2],learning_rate=learning_rate,epochs=epochs,early_stop_delta=early_stop_delta)
            except Exception as e:
                print("found an errror:", e)
        
        return
            ##check if file_naming+"_train_plot.png" already exist
    else:
        train_and_eval_network(raw_data_dir,gen_data_dir=gen_data_dir,out_dir=out_dir,existing_dir=existing_dir,input_size=input_size,batch_size=batch_size,base_model_name=base_model_name,data_part=data_part,val_fraciton=val_fraciton,problem=problem,learning_rate=learning_rate,epochs=epochs,early_stop_delta=early_stop_delta)



def train_and_eval_network(raw_data_dir: pathlib.Path,
    gen_data_dir: pathlib.Path,
    out_dir: pathlib.Path,
    existing_dir: pathlib.Path,
    input_size: int,
    batch_size: int,
    base_model_name: str,
    data_part:str,
    val_fraciton: float,
    problem: str,
    learning_rate: float,
    epochs: int,
    early_stop_delta: float):
    
    file_naming = f"{problem}_{base_model_name}_{data_part}"
    
    print("check if ",(existing_dir/(file_naming+"_train_plot.png")))
    if(existing_dir):
        if os.path.isfile((existing_dir/(file_naming+"_train_plot.png") )):
            print("allready exist")
            return
        #print("check if ",(existing_dir/(file_naming+"_train_plot.png"), exist )
    
    #return



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
    # PRETRAINED_VGG16_WEIGHTS_FILE = (
    #     Path().absolute()
    #     / "pretrained_weights"
    #     / "vgg16_weights_tf_dim_ordering_tf_kernels.h5"
    # )
    # maybe_download_vgg16_pretrained_weights(PRETRAINED_VGG16_WEIGHTS_FILE)


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


    @unique
    class MLProblem(Enum):
        malignancy_prediction = "malignancy"
        nodule_type_prediction = "noduletype"


    # Here you can switch the machine learning problem to solve
    if problem == "noduletype":
        problem =  "noduletype"#MLProblem.noduletype
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
    elif problem ==  "noduletype": # MLProblem.nodule_type_prediction:
        # We made this problem a multiclass classification problem with three classes:
        # 0 - non-solid, 1 - part-solid, 2 - solid
        num_classes = 3
        batch_size = batch_size #30  # make this a factor of three to fit three classes evenly per batch during training
        # This dataset has only few part-solid nodules in the dataset, so we make a tiny validation set
        num_validation_samples = batch_size * 2
        labels = full_dataset["labels_nodule_type"]
        # It is possible to generate training labels yourself using the raw annotations of the radiologists...
        labels_raw = full_dataset["labels_nodule_type_raw"]
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


    def clip_and_scale(
        data: np.ndarray, min_value: float = -1000.0, max_value: float = 400.0
    ) -> np.ndarray:
        data = (data - min_value) / (max_value - min_value)
        data[data > 1] = 1.0
        data[data < 0] = 0.0
        return data


    def random_flip_augmentation(
        input_sample: np.ndarray, axis: Tuple[int, ...] = (1, 2)
    ) -> np.ndarray:
        for ax in axis:
            if np.random.random_sample() > 0.5:
                input_sample = np.flip(input_sample, axis=ax)
        return input_sample


    def shared_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
        """Preprocessing that is used by both the training and validation sets during training

        :param input_batch: np.ndarray [batch_size x channels x dim_x x dim_y]
        :return: np.ndarray preprocessed batch
        """
        input_batch = clip_and_scale(input_batch, min_value=-1000.0, max_value=400.0)
        # Can add more preprocessing here...
        return input_batch


    def train_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
        input_batch = shared_preprocess_fn(input_batch=input_batch)

        output_batch = []
        for sample in input_batch:
            sample = random_flip_augmentation(sample, axis=(1, 2))
            output_batch.append(sample)

        return np.array(output_batch)


    def validation_preprocess_fn(input_batch: np.ndarray) -> np.ndarray:
        input_batch = shared_preprocess_fn(input_batch=input_batch)
        return input_batch

    
    def select_model(base_model_name):
        if base_model_name == 'ResNet50':
                base_model = ResNet50(
                    include_top=False, #define the output of the last layers
                    weights="imagenet",
                    input_tensor=None,
                    input_shape=None,
                    pooling="avg",
                    classes=num_classes,
                    classifier_activation=None,#//softmax
                )
                ## Extended part
                x = layers.Flatten()(base_model.output)
                x = layers.Dense(1024, activation='relu')(x)
                o = layers.Dense(num_classes, activation='sigmoid')(x)

                model = keras.Model(inputs=base_model.input, outputs=o)
                
        if base_model_name == 'ResNet101':
                base_model = ResNet101(
                    include_top=False, #define the output of the last layers
                    weights="imagenet",
                    input_tensor=None,
                    input_shape=None,
                    pooling="avg",
                    classes=num_classes,
                    classifier_activation=None,#//softmax
                )
                ## Extended part
                x = layers.Flatten()(base_model.output)
                x = layers.Dense(1024, activation='relu')(x)
                o = layers.Dense(num_classes, activation='sigmoid')(x)

                model = keras.Model(inputs=base_model.input, outputs=o)
        if base_model_name == 'ResNet152':
                base_model = ResNet152(
                    include_top=False, #define the output of the last layers
                    weights="imagenet",
                    input_tensor=None,
                    input_shape=None,
                    pooling="avg",
                    classes=num_classes,
                    classifier_activation=None,#//softmax
                )
                ## Extended part
                x = layers.Flatten()(base_model.output)
                x = layers.Dense(1024, activation='relu')(x)
                o = layers.Dense(num_classes, activation='sigmoid')(x)

                model = keras.Model(inputs=base_model.input, outputs=o)
        if base_model_name == "vgg16":
            # Define model
            ## We use the VGG16 model
            ## Base model
            model = VGG16(
                    include_top=True,
                    weights=None,
                    input_tensor=None,
                    input_shape=None,
                    pooling=None,
                    classes=num_classes,
                    classifier_activation="softmax",
                )
        if base_model_name == "ffncn":
            # Try for simple network.
            inputs = keras.Input(shape=(None, None, 3))
            processed = keras.layers.RandomCrop(width=32, height=32)(inputs)
            conv = keras.layers.Conv2D(filters=2, kernel_size=3)(processed)
            pooling = keras.layers.GlobalAveragePooling2D()(conv)
            feature = keras.layers.Dense(10)(pooling)

            full_model = keras.Model(inputs, feature)
            backbone = keras.Model(processed, conv)
            activations = keras.Model(conv, feature)
        

        return model


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
    


    #model = keras.Model(inputs=base_model.input, outputs=o)
    model = select_model(base_model_name)

    

    # Show the model layers
    print(model.summary())

    # Load the pretrained imagenet VGG model weights except for the last layer
    # Because the pretrained weights will have a data size mismatch in the last layer of our model
    # two warnings will be raised, but these can be safely ignored.
    # model.load_weights(str(PRETRAINED_VGG16_WEIGHTS_FILE), by_name=True, skip_mismatch=True)

    # Prepare model for training by defining the loss, optimizer, and metrics to use
    # Output labels and predictions are one-hot encoded, so we use the categorical_accuracy metric
    model.compile(
        optimizer=SGD(learning_rate=learning_rate, momentum=0.8, nesterov=True),
        loss=categorical_crossentropy,
        metrics=["categorical_accuracy"],
    )
    
    summary = model.summary()
    print(summary)




    # Start actual training process
    output_model_file = (
        TRAINING_OUTPUT_DIRECTORY / (file_naming+"best_val_accuracy.h5")
    )
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
    )



    # generate a plot using the training history...
    output_history_img_file = (
        TRAINING_OUTPUT_DIRECTORY / (file_naming+"_train_plot.png")
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

    print(history.history.keys())

    output_history_file=( TRAINING_OUTPUT_DIRECTORY / (str(max(history.history['val_categorical_accuracy']))+file_naming+"_hist.json"))
    #str(max(history.history['val_categorical_accuracy']))+
    json.dump(history.history, open(output_history_file, 'w'), indent=4)

if __name__ == "__main__":
    main()
