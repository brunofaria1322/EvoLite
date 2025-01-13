import torch
from typing import Dict, OrderedDict
import torch.nn.utils.prune as prune


if __name__ == "__main__":
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    rate = 0.5

    # NEURON PRUNING 1
    model1 = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    # First pass: Calculate all neuron significance scores
    all_scores = []
    layer_scores = {}

    for name, module in model1.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            weights = module.weight.data
            if isinstance(module, torch.nn.Conv2d):
                weights = weights.view(weights.size(0), -1)

            # Calculate L1-norm for each neuron
            scores = torch.sum(torch.abs(weights), dim=1)
            layer_scores[name] = scores
            all_scores.append(scores)

    # Calculate threshold from all scores
    if all_scores:
        all_scores_tensor = torch.cat(all_scores)
        threshold = torch.quantile(all_scores_tensor, rate)

        # Second pass: Apply pruning
        for name, module in model1.named_modules():
            if name in layer_scores:
                scores = layer_scores[name]
                # Create pruning mask based on threshold
                prune_mask = scores > threshold  # Keep neurons above threshold

                # Apply structured pruning using PyTorch's pruning utility
                if isinstance(module, torch.nn.Linear):
                    prune.custom_from_mask(
                        module,
                        name="weight",
                        mask=prune_mask.unsqueeze(1).expand_as(module.weight),
                    )
                    if module.bias is not None:
                        prune.custom_from_mask(module, name="bias", mask=prune_mask)
                elif isinstance(module, torch.nn.Conv2d):
                    mask_shape = prune_mask.view(-1, 1, 1, 1).expand_as(module.weight)
                    prune.custom_from_mask(module, name="weight", mask=mask_shape)
                    if module.bias is not None:
                        prune.custom_from_mask(module, name="bias", mask=prune_mask)

                # Make pruning permanent
                prune.remove(module, "weight")
                if module.bias is not None:
                    prune.remove(module, "bias")

    # print model neurons and architecture
    print(model1)
    # count number of neurons
    num_neurons = 0
    for module in model1.modules():
        if isinstance(module, torch.nn.Linear):
            num_neurons += module.weight.shape[1]
    print(num_neurons)

    # NEURON PRUNING 2
    model2 = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    # First pass: Calculate all neuron significance scores
    all_scores = []
    layer_scores = {}

    for name, module in model2.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            weights = module.weight.data
            if isinstance(module, torch.nn.Conv2d):
                weights = weights.view(weights.size(0), -1)

            # Calculate L1-norm for each neuron
            scores = torch.sum(torch.abs(weights), dim=1)
            layer_scores[name] = scores
            all_scores.append(scores)

    # Calculate threshold from all scores
    if all_scores:
        all_scores_tensor = torch.cat(all_scores)
        threshold = torch.quantile(all_scores_tensor, rate)

        # Second pass: Prune neurons below threshold
        with torch.no_grad():
            for name, module in model2.named_modules():
                if name in layer_scores:
                    scores = layer_scores[name]
                    prune_mask = scores <= threshold

                    if isinstance(module, torch.nn.Linear):
                        module.weight.data[prune_mask, :] = 0
                        if module.bias is not None:
                            module.bias.data[prune_mask] = 0
                    elif isinstance(module, torch.nn.Conv2d):
                        module.weight.data[prune_mask, :, :, :] = 0
                        if module.bias is not None:
                            module.bias.data[prune_mask] = 0

    # print model neurons and architecture
    print(model2)
    # count number of neurons
    num_neurons = 0
    for module in model2.modules():
        if isinstance(module, torch.nn.Linear):
            num_neurons += module.weight.shape[1]
    print(num_neurons)

    # NEURON PRUNING 3
    model3 = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    new_model = type(model3)()

    with torch.no_grad():
        for name, module in model3.named_modules():
            if isinstance(module, torch.nn.Linear):
                weights = module.weight.data
                # Calculate L1-norm for each neuron
                scores = torch.sum(torch.abs(weights), dim=1)
                num_neurons = len(scores)
                keep_neurons = int(num_neurons * (1 - rate))

                # Get indices of top neurons to keep
                top_indices = torch.argsort(scores, descending=True)[:keep_neurons]

                # Create new smaller layer
                new_layer = torch.nn.Linear(module.in_features, keep_neurons)

                # Copy weights and bias for kept neurons
                new_layer.weight.data = module.weight.data[top_indices]
                if module.bias is not None:
                    new_layer.bias.data = module.bias.data[top_indices]

                # Replace layer in new model
                setattr(new_model, name, new_layer)

    # print model neurons and architecture
    print(new_model)
    # count number of neurons
    num_neurons = 0
    for module in new_model.modules():
        if isinstance(module, torch.nn.Linear):
            num_neurons += module.weight.shape[1]
    print(num_neurons)

    # NEURON PRUNING 4
    new_model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    # Get ordered dict of all layers
    layers: OrderedDict[str, torch.nn.Module] = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            layers[name] = module

    # Store neuron indices to keep for each layer
    keep_indices: Dict[str, torch.Tensor] = {}

    # First pass: determine which neurons to keep
    with torch.no_grad():
        for name, module in layers.items():
            weights = module.weight.data
            if isinstance(module, torch.nn.Conv2d):
                # For Conv2d, treat each filter as a neuron
                scores = torch.sum(torch.abs(weights.view(weights.size(0), -1)), dim=1)
            else:
                scores = torch.sum(torch.abs(weights), dim=1)

            num_neurons = len(scores)
            keep_neurons = int(num_neurons * (1 - rate))
            keep_neurons = max(1, keep_neurons)  # Always keep at least one neuron

            # Get indices of top neurons to keep
            keep_indices[name] = torch.argsort(scores, descending=True)[:keep_neurons]

    # Second pass: create and connect new layers
    layer_names = list(layers.keys())
    for i, (name, module) in enumerate(layers.items()):
        if isinstance(module, torch.nn.Linear):
            in_features = module.in_features
            out_features = len(keep_indices[name])

            # Adjust input features if not first layer
            if i > 0:
                prev_name = layer_names[i - 1]
                prev_indices = keep_indices[prev_name]
                if isinstance(layers[prev_name], torch.nn.Conv2d):
                    # Calculate flattened size from previous conv layer
                    prev_module = layers[prev_name]
                    prev_output = prev_module.out_channels
                    in_features = (
                        prev_output
                        * prev_module.kernel_size[0]
                        * prev_module.kernel_size[1]
                        * len(prev_indices)
                    )
                else:
                    in_features = len(prev_indices)

            # Create new layer
            new_layer = torch.nn.Linear(in_features, out_features)

            # Copy weights and bias for kept neurons
            if i > 0 and isinstance(layers[layer_names[i - 1]], torch.nn.Linear):
                new_layer.weight.data = module.weight.data[keep_indices[name]][
                    :, keep_indices[layer_names[i - 1]]
                ]
            else:
                new_layer.weight.data = module.weight.data[keep_indices[name]]

            if module.bias is not None:
                new_layer.bias.data = module.bias.data[keep_indices[name]]

            # Replace layer in new model
            setattr(new_model, name, new_layer)

        elif isinstance(module, torch.nn.Conv2d):
            in_channels = module.in_channels
            out_channels = len(keep_indices[name])

            # Adjust input channels if not first layer
            if i > 0:
                prev_name = layer_names[i - 1]
                if isinstance(layers[prev_name], torch.nn.Conv2d):
                    in_channels = len(keep_indices[prev_name])

            # Create new layer
            new_layer = torch.nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=1,  # Reset groups as it might not be valid after pruning
                bias=module.bias is not None,
            )

            # Copy weights and bias for kept filters
            if i > 0 and isinstance(layers[layer_names[i - 1]], torch.nn.Conv2d):
                new_layer.weight.data = module.weight.data[keep_indices[name]][
                    :, keep_indices[layer_names[i - 1]], :, :
                ]
            else:
                new_layer.weight.data = module.weight.data[keep_indices[name]]

            if module.bias is not None:
                new_layer.bias.data = module.bias.data[keep_indices[name]]

            # Replace layer in new model
            setattr(new_model, name, new_layer)

    # print model neurons and architecture
    print(new_model)
    # count number of neurons
    num_neurons = 0
    for module in new_model.modules():
        if isinstance(module, torch.nn.Linear):
            num_neurons += module.weight.shape[1]
    print(num_neurons)
