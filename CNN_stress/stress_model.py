import toml
from pathlib import Path
from typing import Callable, Optional, Union, Iterable
import blobconverter
import shutil
import json

import wandb

from tensorflow.keras.applications.resnet import (
    ResNet50,
    ResNet101,
    ResNet152,
)

from tensorflow.keras.applications.resnet_v2 import (
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
)

from tensorflow.keras.applications.mobilenet import MobileNet

from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.metrics import (
    BinaryAccuracy,
    CategoricalAccuracy,
    Precision,
    Recall,
)
from tensorflow.keras.backend import epsilon
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential, load_model, clone_model
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam, SGD, Adamax, RMSprop

from tensorflow.keras.models import load_model
import typer

import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import (
    convert_variables_to_constants_v2,
)

import tensorflow as tf

from tensorflow.python.keras import backend as K

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.InteractiveSession(config=config)
K.set_session(tf.compat.v1.Session(config=config))

tf.config.run_functions_eagerly(True)

import typer

MODEL_TYPES = {
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNet50V2": ResNet50V2,
    "ResNet101V2": ResNet101V2,
    "ResNet152V2": ResNet152V2,
    "MobileNet": MobileNet,
}


def F1(y_true, y_pred):
    prec_fun = Precision()
    rec_fun = Recall()

    precision = prec_fun(y_true, y_pred)
    recall = rec_fun(y_true, y_pred)

    return 2 * ((precision * recall) / (precision + recall + epsilon()))


class CNN:
    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        base_model: Optional[str] = None,
    ):
        """Initialize a new CNN model.

        Parameters
        ----------
        model_path : Optional[Union[str, Path]]
            Path to model, by default None
            If it is `None`, a `ResNet50v2` with weights of `imagenet` is loaded.
        base_model : Optional[str], optional
            Base model to load weights on it, by default None.
            If it is `None`, a local saved model will be loaded using `model_path`.
        """
        if model_path is not None:
            self.load(model_path, base_model)
        else:
            self.load()

    def train(
        self,
        train_images_folder: Union[Path, str],
        val_images_folder: Union[Path, str],
        labels_names: Union[str, Iterable[str]] = "inferred",
        n_labels: Optional[int] = 2,
        color_mode: str = "rgb",
        image_size: Iterable[int] = (224, 224),
        shuffle: bool = True,
        batch_size: int = 32,
        optimizer_name: str = "Adam",
        learning_rate: float = 1e-5,
        epochs: int = 10,
        metrics: Iterable[str] = ("F1", "binary_accuracy", "precision", "recall"),
        early_stopping_metric: str = "val_F1",
        early_stopping_patience: str = 5,
        early_stopping_mode: str = "max",
        callbacks: list[Callable] = [],
    ):
        if n_labels is None:
            assert (
                not labels_names == "inferred"
            ), f"If labels are inferred `n_labels` must be passed."
            n_labels = len(labels_names)

        layer_names = [layer.name for layer in self.MODEL.layers]

        if n_labels > 2:
            label_mode = "int"
            loss_function = "sparse_categorical_crossentropy"
            if "final_layer" not in layer_names:
                self.MODEL.add(
                    Dense(n_labels, activation="softmax", name="final_layer")
                )
        else:
            label_mode = "binary"
            loss_function = "binary_crossentropy"
            if "final_layer" not in layer_names:
                self.MODEL.add(Dense(1, activation="sigmoid", name="final_layer"))

        train_dataset = image_dataset_from_directory(
            train_images_folder,
            labels=labels_names,
            batch_size=batch_size,
            color_mode=color_mode,
            image_size=image_size,
            shuffle=shuffle,
            label_mode=label_mode,
        )

        validation_dataset = image_dataset_from_directory(
            val_images_folder,
            labels=labels_names,
            batch_size=batch_size,
            color_mode=color_mode,
            image_size=image_size,
            shuffle=shuffle,
            label_mode=label_mode,
        )

        metrics_list = []

        for m in metrics:
            if m == "binary_accuracy":
                if label_mode == "binary":
                    metrics_list.append(BinaryAccuracy())
                else:
                    metrics_list.append(CategoricalAccuracy())
            elif m == "precision":
                metrics_list.append(Precision())
            elif m == "recall":
                metrics_list.append(Recall())
            elif m == "F1":
                metrics_list.append(F1)

        if optimizer_name == "Adam":
            optimizer = Adam(
                learning_rate=learning_rate,
            )
        elif optimizer_name == "SGD":
            optimizer = SGD(
                learning_rate=learning_rate,
            )
        elif optimizer_name == "Adamax":
            optimizer = Adamax(
                learning_rate=learning_rate,
            )
        elif optimizer_name == "RMSprop":
            optimizer = RMSprop(
                learning_rate=learning_rate,
            )

        self.MODEL.compile(
            optimizer=optimizer, loss=loss_function, metrics=metrics_list
        )

        early_stopping_monitor = EarlyStopping(
            monitor=early_stopping_metric,
            patience=early_stopping_patience,
            mode=early_stopping_mode,
            restore_best_weights=True,
        )

        callbacks.append(early_stopping_monitor)

        training_history = self.MODEL.fit(
            train_dataset,
            validation_data=validation_dataset,
            epochs=epochs,
            callbacks=[callbacks],
        )

    def _train_wandb(self):
        aux_model = clone_model(self.MODEL)

        run = wandb.init(name=f"run_{self.run_number}")
        config = dict(run.config)

        keras_callback = wandb.keras.WandbCallback()

        self.train(
            train_images_folder=self.train_images_folder,
            val_images_folder=self.val_images_folder,
            callbacks=[keras_callback],
            **config,
        )

        self.save(f"{self.wandb_runs_dir}/{run.name}.h5")

        self.MODEL = aux_model

        shutil.rmtree("/home/users/ucadatalab_group/javierj/Baby-Stress/wandb")

        self.run_number += 1

    def hiperparameters_search_wandb(
        self,
        wandb_project: str,
        number_runs: int,
        wandb_runs_dir: str,
        hiperparams_search_config: dict[str, str],
        train_images_folder: Union[Path, str],
        val_images_folder: Union[Path, str],
    ):
        self.run_number = 0

        self.wandb_runs_dir = wandb_runs_dir

        self.train_images_folder = train_images_folder
        self.val_images_folder = val_images_folder

        wandb.login()
        sweep_id = wandb.sweep(hiperparams_search_config, project=wandb_project)
        wandb.agent(
            sweep_id=sweep_id,
            project=wandb_project,
            function=self._train_wandb,
            count=number_runs,
        )

    def save(self, model_path: Union[str, Path]):
        """Save the weights of the model.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the file where the model will be saved. Must be `.h5` extension.
        """

        self.MODEL.save(model_path)

    def load(
        self,
        model_path: Union[str, Path] = "imagenet",
        base_model: Optional[str] = "ResNet50V2",
    ):
        """Load the architecture and the weights of a model.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the model saved.
        base_model : Optional[str], optional
            Base model to load weights on it, by default None.
            If it is `None`, a local saved model will be loaded using `model_path`.
        """

        if base_model is None:
            self.MODEL = load_model(model_path, custom_objects={"F1": F1})

        else:
            self.MODEL = Sequential()
            model = MODEL_TYPES[base_model](weights=model_path)
            self.MODEL.add(model)

    def to_pb(self, path_out_folder):
        # path of the directory where you want to save your model

        frozen_graph_filename = "frozen_graph"
        model = self.MODEL
        full_model = tf.function(lambda x: model(x))
        full_model = full_model.get_concrete_function(
            tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype)
        )  # Get frozen ConcreteFunction
        frozen_func = convert_variables_to_constants_v2(full_model)
        frozen_func.graph.as_graph_def()
        layers = [op.name for op in frozen_func.graph.get_operations()]
        print("-" * 60)
        print("Frozen model layers: ")

        for layer in layers:
            print(layer)

        print("-" * 60)
        print("Frozen model inputs: ")
        print(frozen_func.inputs)
        print("Frozen model outputs: ")
        print(frozen_func.outputs)  # Save frozen graph to disk
        tf.io.write_graph(
            graph_or_graph_def=frozen_func.graph,
            logdir=path_out_folder,
            name=f"{frozen_graph_filename}.pb",
            as_text=False,
        )

    def to_blob(self, path_out_folder):
        self.to_pb(path_out_folder)

        blob_path = blobconverter.from_tf(
            frozen_pb=f"{path_out_folder}/frozen_graph.pb",
            data_type="FP16",
            shaves=5,
            version=blobconverter.Versions.v2020_1,
            optimizer_params=[
                "--reverse_input_channels",
                "--input_shape=[1,224,224,3]",
            ],
        )

        shutil.move(blob_path, f"{path_out_folder}/depthai_model.blob")


def main(config_path: Optional[str] = "./CNN_stress/config.toml"):

    config_dict = toml.load(config_path)

    hyperparam_path = config_dict["wandb"]["hyperparam_search_config_path"]

    with open(hyperparam_path, "r") as f:
        hyperparam_config = json.load(f)

    model = CNN(**config_dict["load"])
    model.hiperparameters_search_wandb(
        "Baby_stress_CNN",
        20,
        config_dict["wandb"]["save_dir"],
        hyperparam_config,
        config_dict["wandb"]["train_images_folder"],
        config_dict["wandb"]["val_images_folder"],
    )

    # model = CNN(**config_dict["load"])
    # model.train(**config_dict["training"])
    # model.save(**config_dict["save"])
    # model.load(config_dict["save"]["model_path"], base_model=None)


if __name__ == "__main__":
    typer.run(main)
