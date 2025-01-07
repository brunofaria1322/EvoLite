from typing import Any
import torch
import tensorflow as tf


def validate_model(model: torch.nn.Module | tf.keras.Model) -> None:
    """
    Validate if the model is PyTorch or TensorFlow.

    Args:
        model: The model to validate.

    Raises:
        ValueError: If the model is not supported.
    """
    if not (isinstance(model, torch.nn.Module) or isinstance(model, tf.keras.Model)):
        raise ValueError("Model must be a PyTorch or TensorFlow model.")


def validate_dataset(dataset: Any) -> None:
    """
    Validate if the dataset is in the expected format.

    Args:
        dataset: The dataset to validate.

    Raises:
        ValueError: If the dataset is invalid.
    """
    if not (hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__")):
        raise ValueError("Dataset must support indexing and have a length.")
