from typing import Literal
import torch
import torch.nn.utils.prune as prune


class GlobalStructuredPruning(prune.BasePruningMethod):
    """
    Custom global structured pruning method.
    Prunes entire neurons/filters based on their L2 norm across the entire model.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount):
        self.amount = amount

    def compute_mask(self, t: torch.Tensor, default_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute the mask for global structured pruning.

        Args:
            t: The tensor to prune.
            default_mask: The default mask (unused here).

        Returns:
            The pruning mask.
        """
        # Start with the default mask (no pruning)
        mask = default_mask.clone()

        # Calculate the L2 norm of each neuron/filter
        if len(t.shape) == 4:  # Convolutional layer (C_out, C_in, H, W)
            norm = torch.norm(t, p=2, dim=(1, 2, 3))  # L2 norm along input channels and spatial dimensions
        elif len(t.shape) == 2:  # Fully connected layer (out_features, in_features)
            norm = torch.norm(t, p=2, dim=1)  # L2 norm along input features
        else:
            raise ValueError(f"Unsupported tensor shape for structured pruning: {t.shape}")

        # Calculate the global threshold
        threshold = torch.quantile(norm, self.amount)

        # Update the mask based on the threshold
        mask[norm <= threshold] = 0

        return mask


def global_structured_prune(module, name, amount):
    """
    Apply global structured pruning to a module.

    Args:
        module: The module to prune.
        name: The name of the parameter to prune (e.g., "weight").
        amount: The fraction of neurons/filters to prune.
    """
    GlobalStructuredPruning.apply(module, name, amount)


class PyTorchPruning:
    def __init__(self, prune_type: Literal["structured", "unstructured"], rate: float):
        """
        Initialize the pruning module for PyTorch.

        Args:
            prune_type: Type of pruning ("structured" or "unstructured")
            rate: Fraction of weights/neurons to prune (e.g., 0.5 for 50%).
        """
        self.prune_type = prune_type
        self.rate = rate

    def apply(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply pruning to the PyTorch model.

        Args:
            model: The PyTorch model to prune.

        Returns:
            The pruned PyTorch model.
        """
        if self.prune_type == "unstructured":
            return self._prune_unstructured(model)
        elif self.prune_type == "structured":
            return self._prune_structured(model)
        else:
            raise ValueError("Unsupported prune_type.")

    def _prune_unstructured(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply global unstructured pruning (L1 norm) to the model.

        Args:
            model: The PyTorch model to prune.

        Returns:
            The pruned PyTorch model.
        """
        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        ]
        prune.global_unstructured(
            parameters_to_prune, pruning_method=prune.L1Unstructured, amount=self.rate
        )
        return model

    def _prune_structured(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        Apply global structured pruning (L2 norm) to the model.

        Args:
            model: The PyTorch model to prune.

        Returns:
            The pruned PyTorch model.
        """
        # Collect all parameters to prune
        parameters_to_prune = [
            (module, "weight")
            for module in model.modules()
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d))
        ]

        # Apply global structured pruning
        for module, name in parameters_to_prune:
            global_structured_prune(module, name, self.rate)

        return model
