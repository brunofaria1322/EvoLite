import torch
import tensorflow as tf


def save_model(model: torch.nn.Module | tf.keras.Model, path: str) -> None:
    """
    Save a PyTorch or TensorFlow model to a file.

    Args:
        model: The model to save.
        path: Path to save the model.
    """
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), path)
    elif isinstance(model, tf.keras.Model):
        model.save_weights(path)
    else:
        raise ValueError("Unsupported model type.")


def load_model(model: torch.nn.Module | tf.keras.Model, path: str) -> None:
    """
    Load a PyTorch or TensorFlow model from a file.

    Args:
        model: The model to load weights into.
        path: Path to the saved model.
    """
    if isinstance(model, torch.nn.Module):
        model.load_state_dict(torch.load(path, weights_only=True))
    elif isinstance(model, tf.keras.Model):
        model.load_weights(path)
    else:
        raise ValueError("Unsupported model type.")
