import torch
import tensorflow as tf
from src.evolite.compression.pruning import Pruning


# PyTorch
def test_pytorch_weight_pruning():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    pruning = Pruning(target="weights", rate=0.5)
    pruned_model = pruning.apply(model)
    assert isinstance(pruned_model, torch.nn.Module)

    # TODO: Verify
    # Count the number of zeros in the weights of the pruned model and compare it with the expected number of zeros
    num_zeros = 0
    num_total = 0
    for module in pruned_model.modules():
        if isinstance(module, torch.nn.Linear):
            num_zeros += torch.sum(module.weight.data == 0).item()
            num_total += module.weight.numel()

    assert num_zeros == 0.5 * num_total


def test_pytorch_neuron_pruning():
    model = torch.nn.Sequential(
        torch.nn.Linear(10, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    pruning = Pruning(target="neurons", rate=0.5)
    try:
        pruning.apply(model)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass


# TensorFlow
def test_tensorflow_weight_pruning():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(2),
        ]
    )
    pruning = Pruning(target="weights", rate=0.5)
    pruned_model = pruning.apply(model)
    assert isinstance(pruned_model, tf.keras.Model)

    # TODO: Verify
    # Count the number of zeros in the weights of the pruned model and compare it with the expected number of zeros
    num_zeros = 0
    num_total = 0
    for layer in pruned_model.layers:
        if isinstance(layer, tf.keras.layers.Dense):
            num_zeros += tf.reduce_sum(tf.cast(layer.get_weights()[0] == 0, tf.int32))
            num_total += tf.size(layer.get_weights()[0])

    num_zeros = num_zeros.numpy()
    num_total = num_total.numpy()

    assert num_zeros == 0.5 * num_total


def test_tensorflow_neuron_pruning():
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(10,)),
            tf.keras.layers.Dense(5),
            tf.keras.layers.ReLU(),
            tf.keras.layers.Dense(2),
        ]
    )
    pruning = Pruning(target="neurons", rate=0.5)
    try:
        pruning.apply(model)
        assert False, "Expected NotImplementedError"
    except NotImplementedError:
        pass
