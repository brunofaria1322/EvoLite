import pytest
import numpy as np
from evolite.models.neural_net import CompressibleNetwork


def test_network_initialization():
    """Test proper network initialization."""
    network = CompressibleNetwork()
    assert len(network.layers) == 0
    assert network.compression_ratio == 1.0


def test_network_evaluation(sample_network, sample_dataset):
    """Test network forward pass."""
    X, _ = sample_dataset
    output = sample_network.evaluate(X)

    assert output.shape[0] == X.shape[0]
    assert output.shape[1] == sample_network.layers[-1]["weights"].shape[1]


def test_layer_compression():
    """Test compression of individual layers."""
    network = CompressibleNetwork()
    network.layers = [{"type": "dense", "weights": np.random.randn(10, 10)}]

    network.compress_layer(0, method="pruning", threshold=0.5)
    assert (
        np.sum(network.layers[0]["weights"] != 0) < 100
    )  # Some weights should be zero
