import torch
from evolite.compression.pytorch.pruning import PyTorchPruning
from ..utils import LeNet


def test_pytorch_pruning_unstructured():
    model = LeNet()
    pruning = PyTorchPruning(prune_type="unstructured", rate=0.5)
    pruned_model = pruning.apply(model)
    assert isinstance(pruned_model, torch.nn.Module)

    # Check if weights were pruned
    num_weights = 0
    num_pruned = 0
    for module in pruned_model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            num_weights += module.weight.numel()
            num_pruned += torch.sum(module.weight == 0).item()

    assert num_pruned == 0.5 * num_weights  # Check if 50% weights were pruned


def test_pytorch_pruning_structured():
    model = LeNet()
    pruning = PyTorchPruning(prune_type="structured", rate=0.5)
    pruned_model = pruning.apply(model)
    assert isinstance(pruned_model, torch.nn.Module)

    # Check if weights were pruned
    num_structures = 0
    num_pruned = 0
    for module in pruned_model.modules():
        if isinstance(module, torch.nn.Linear) or isinstance(module, torch.nn.Conv2d):
            num_structures += module.weight.shape[0]

            if len(module.weight.shape) == 4:  # Convolutional layer (C_out, C_in, H, W)
                num_pruned += torch.sum(
                    torch.all(module.weight == 0, dim=(1, 2, 3))
                ).item()
            elif (
                len(module.weight.shape) == 2
            ):  # Fully connected layer (out_features, in_features)
                num_pruned += torch.sum(torch.all(module.weight == 0, dim=1)).item()
            else:
                raise ValueError(
                    f"Unsupported tensor shape for structured pruning: {module.weight.shape}"
                )

    assert num_pruned == 0.5 * num_structures  # Check if 50% structures were pruned
