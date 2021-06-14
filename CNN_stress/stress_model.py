from typing import Optional, Union
from pathlib import Path
from keras.applications.resnet50 import (
    ResNet50,
    ResNet101,
    ResNet152,
    ResNet50V2,
    ResNet101V2,
    ResNet152V2,
)

MODEL_TYPES = {
    "ResNet50": ResNet50,
    "ResNet101": ResNet101,
    "ResNet152": ResNet152,
    "ResNet50V2": ResNet50V2,
    "ResNet101V2": ResNet101V2,
    "ResNet152V2": ResNet152V2,
}


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
            self.MODEL = self.load(model_path, base_model)
        else:
            self.MODEL = self.load()

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
        if base_model is None:
            with open(f"{model_path}/model_arquitecture.json", "r") as f:
                self.MODEL = f.read()

            self.MODEL.load_weights(f"{model_path}/model_weights.h5")

        else:
            self.MODEL = MODEL_TYPES[base_model](weights=model_path)
