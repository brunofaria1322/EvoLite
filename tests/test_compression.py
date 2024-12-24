import pytest
import numpy as np
from evolite.core.compression import compress_network
from evolite.models.neural_net import CompressibleNetwork


def test_compression_ratio(sample_network):
    """Test that compression actually reduces network size."""
    original_size = sum(layer["weights"].size for layer in sample_network.layers)

    compressed = compress_network(sample_network, target_size=0.5)
    compressed_size = sum(layer["weights"].size for layer in compressed.layers)

    assert compressed_size < original_size
    assert compressed.compression_ratio <= 0.5


def test_compression_maintains_input_output_dims(sample_network):
    """Test that compression preserves network input/output dimensions."""
    compressed = compress_network(sample_network, target_size=0.5)

    assert (
        compressed.layers[0]["weights"].shape[0]
        == sample_network.layers[0]["weights"].shape[0]
    )
    assert (
        compressed.layers[-1]["weights"].shape[1]
        == sample_network.layers[-1]["weights"].shape[1]
    )


def test_invalid_compression_ratio():
    """Test handling of invalid compression ratios."""
    network = CompressibleNetwork()

    with pytest.raises(ValueError):
        compress_network(network, target_size=1.5)
    with pytest.raises(ValueError):
        compress_network(network, target_size=-0.1)
