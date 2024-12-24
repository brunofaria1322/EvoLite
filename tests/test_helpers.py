import pytest
import numpy as np
from evolite.utils.helpers import calculate_compression_ratio, evaluate_fitness


def test_compression_ratio_calculation(sample_network):
    """Test accurate compression ratio calculation."""
    ratio = calculate_compression_ratio(sample_network)
    assert 0 <= ratio <= 1

    # Modify network and check ratio updates
    sample_network.layers[0]["weights"] *= 0  # Set some weights to zero
    new_ratio = calculate_compression_ratio(sample_network)
    assert new_ratio < ratio


def test_fitness_evaluation(sample_network, sample_dataset):
    """Test fitness evaluation combines accuracy and compression."""
    X, y = sample_dataset

    # Test with different compression weights
    fitness1 = evaluate_fitness(sample_network, X, y, compression_weight=0.2)
    fitness2 = evaluate_fitness(sample_network, X, y, compression_weight=0.8)

    assert 0 <= fitness1 <= 1
    assert 0 <= fitness2 <= 1
    assert fitness1 != fitness2  # Different weights should yield different scores
