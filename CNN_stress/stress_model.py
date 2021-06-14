from typing import Union
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
        self, model_path: Union[str, Path] = None, base_model: Union[str, None] = None
    ):
        if model_path is not None:
            self.MODEL = self.load(model_path, base_model)
        else:
            self.MODEL = self.load()

    def save(self, model_path: Union[str, Path] = "imagenet"):
        model_json = self.MODEL.to_json()

        with open(f"{model_path}.json", "w") as json_file:
            json_file.write(model_json)

        self.MODEL.save_weights(f"{model_path}.h5")

    def load(
        self,
        model_path: Union[str, Path] = "imagenet",
        base_model: Union[str, None] = "ResNet50v2",
    ):
        if base_model is None:
            with open(f"{model_path}.json", "r") as f:
                self.MODEL = f.read()

            self.MODEL.load_weights(f"{model_path}.h5")

        else:
            self.MODEL = MODEL_TYPES[base_model](weights=model_path)
