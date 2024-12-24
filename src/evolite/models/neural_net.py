from typing import List, Dict, Any
import numpy as np


class CompressibleNetwork:
    """Base class for neural networks that can be compressed."""

    def __init__(self):
        self.layers: List[Dict[str, Any]] = []
        self.compression_ratio: float = 1.0

    def compress_layer(self, layer_idx: int, method: str, **kwargs):
        """
        Apply compression to a specific layer.

        Args:
            layer_idx: Index of layer to compress
            method: Compression method to apply
            **kwargs: Additional compression parameters
        """
        raise NotImplementedError("Coming soon!")

    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        Evaluate the network on input data.

        Args:
            X: Input data

        Returns:
            np.ndarray: Network predictions
        """
        raise NotImplementedError("Coming soon!")
