from typing import Any
import torch
import tensorflow as tf
from ..utils.model_utils import save_model, load_model
from ..utils.validation import validate_model, validate_dataset


class EvoLite:
    def __init__(self, model: torch.nn.Module | tf.keras.Model, dataset: Any) -> None:
        """
        Initialize EvoLite with a model and dataset.

        Args:
            model: A PyTorch or TensorFlow model.
            dataset: A dataset for validation/testing.
        """
        validate_model(model)
        validate_dataset(dataset)
        self.model = model
        self.dataset = dataset

    def compress(self) -> torch.nn.Module | tf.keras.Model:
        """
        Run the compression pipeline.

        Returns:
            Compressed model.
        """
        raise NotImplementedError("compress() not implemented yet.")

    def save(self, path: str) -> None:
        """
        Save the compressed model to a file.

        Args:
            path: Path to save the model.
        """
        save_model(self.model, path)

    def load(self, path: str) -> None:
        """
        Load a compressed model from a file.

        Args:
            path: Path to the saved model.
        """
        load_model(self.model, path)
