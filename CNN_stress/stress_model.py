import toml
from pathlib import Path
from typing import Optional, Union, Iterable

from keras.applications.resnet import (
    ResNet50,
    ResNet101,
    ResNet152,
)

from keras.applications.resnet_v2 import (
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
)
from keras.preprocessing.image import image_dataset_from_directory
from keras.metrics import BinaryAccuracy, CategoricalAccuracy, Precision, Recall
from keras.backend import epsilon
from keras.layers import Dense
from keras.models import Sequential

import typer

SEED = 1234

MODEL_TYPES = {
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNet50V2": ResNet50V2,
    "ResNet101V2": ResNet101V2,
    "ResNet152V2": ResNet152V2,
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
            Path to model save folder, by default None
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
        data_images_folder: Union[Path, str],
        labels_names: Union[str, Iterable[str]] = "inferred",
        n_labels: Optional[int] = None,
        color_mode: str = "rgb",
        image_size: Iterable[int] = (256, 256),
        shuffle: bool = True,
        validation_split: float = 0.2,
        batch_size: int = 32,
        optimizer: str = "adam",
        epochs: int = 10,
        metrics: Iterable[str] = ("acc", "precision", "recall", "f1"),
    ):
        if n_labels is None:
            assert (
                not labels_names == "inferred"
            ), f"If labels are inferred `n_labels` must be passed."
            n_labels = len(labels_names)

        if n_labels > 2:
            label_mode = "int"
            loss_function = "sparse_categorical_crossentropy"
            self.MODEL.add(Dense(n_labels, activation="softmax"))
        else:
            label_mode = "binary"
            loss_function = "binary_crossentropy"
            self.MODEL.add(Dense(1, activation="sigmoid"))

        train_dataset = image_dataset_from_directory(
            data_images_folder,
            labels=labels_names,
            batch_size=batch_size,
            color_mode=color_mode,
            image_size=image_size,
            shuffle=shuffle,
            label_mode=label_mode,
            validation_split=validation_split,
            subset="training",
            seed=SEED,
        )

        validation_dataset = image_dataset_from_directory(
            data_images_folder,
            labels=labels_names,
            batch_size=batch_size,
            color_mode=color_mode,
            image_size=image_size,
            shuffle=shuffle,
            label_mode=label_mode,
            validation_split=validation_split,
            subset="validation",
            seed=SEED,
        )

        metrics_list = []

        for m in metrics:
            if m == "acc":
                if label_mode == "binary":
                    metrics_list.append(BinaryAccuracy())
                else:
                    metrics_list.append(CategoricalAccuracy())
            elif m == "precision":
                metrics_list.append(Precision())
            elif m == "recall":
                metrics_list.append(Recall())
            elif m == "f1":
                metrics_list.append(F1)

        self.MODEL.compile(
            optimizer=optimizer, loss=loss_function, metrics=metrics_list
        )

        self.MODEL.fit(train_dataset, validation_data=validation_dataset, epochs=epochs)

    def save(self, model_path: Union[str, Path]):
        """Save the architecture and the weights of the model.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the folder where the model will be saved.
        """
        model_json = self.MODEL.to_json()

        with open(f"{model_path}/model_arquitecture.json", "w") as json_file:
            json_file.write(model_json)

        self.MODEL.save_weights(f"{model_path}/model_weights.h5")

    def load(
        self,
        model_path: Union[str, Path] = "imagenet",
        base_model: Optional[str] = "ResNet50v2",
    ):
        """Load the architecture and the weights of a model.

        Parameters
        ----------
        model_path : Union[str, Path]
            Path to the folder where the model is saved.
        base_model : Optional[str], optional
            Base model to load weights on it, by default None.
            If it is `None`, a local saved model will be loaded using `model_path`.
        """
        self.MODEL = Sequential()
        if base_model is None:
            with open(f"{model_path}/model_arquitecture.json", "r") as f:
                model = f.read()

            model.load_weights(f"{model_path}/model_weights.h5")

        else:
            model = MODEL_TYPES[base_model](weights=model_path)

        self.MODEL.add(model)


def main(config_path: Optional[str] = "./CNN_stress/config.toml"):

    config_dict = toml.load(config_path)

    model = CNN(**config_dict["load"])
    model.train(**config_dict["training"])
    model.save(**config_dict["save"])


if __name__ == "__main__":
    typer.run(main)
