from typing import Any, Literal
import torch
import tensorflow as tf
from ..utils.model_utils import save_model, load_model
from ..utils.validation import validate_model, validate_dataset
from ..compression.pruning import Pruning


class EvoLite:
    def __init__(self, model: torch.nn.Module | tf.keras.Model, dataset: Any) -> None:
        """
        Initialize EvoLite with a model and dataset.

        Args:
            model: A PyTorch or TensorFlow model.
            dataset: A dataset for validation/testing.
        """
        framework = validate_model(model)
        validate_dataset(dataset)
        self.model = model
        self.dataset = dataset
        self.framework = framework

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

    def prune(self, target: Literal["weights", "neurons"], rate: float) -> None:
        """
        Prune the model.

        Args:
            target: The pruning target ("weights" or "neurons").
            rate: Fraction of weights/neurons to prune.
        """
        pruning = Pruning(target=target, rate=rate)
        self.model = pruning.apply(self.model, self.framework)
