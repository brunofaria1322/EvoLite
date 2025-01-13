from typing import Literal
import torch
from .pytorch.pruning import PyTorchPruning


class Pruning:
    def __init__(
        self,
        prune_type: Literal["structured", "unstructured"],
        rate: float,
    ):
        """
        Initialize the pruning module.

        Args:
            prune_type: Type of pruning ("structured" or "unstructured").
            rate: Fraction of weights/neurons to prune (e.g., 0.5 for 50%).
        """
        if prune_type not in ["structured", "unstructured"]:
            raise ValueError("Unsupported prune_type. Use 'weights' or 'neurons'.")
        self.prune_type = prune_type
        self.rate = rate

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply pruning to the model.

        Args:
            model: The model to prune.

        Returns:
            The pruned model.
        """
        if isinstance(model, torch.nn.Module):
            pruning = PyTorchPruning(self.prune_type, self.rate)
            return pruning.apply(model)
        else:
            raise ValueError(
                "Unsupported framework. Currently only PyTorch is supported."
            )
