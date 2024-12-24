import pytest
import numpy as np
from evolite.models.neural_net import CompressibleNetwork


@pytest.fixture
def sample_network():
    """Fixture providing a simple neural network for testing."""
    network = CompressibleNetwork()
    # Add some basic layers for testing
    network.layers = [
        {
            "type": "dense",
            "weights": np.random.randn(10, 20),
            "bias": np.random.randn(20),
        },
        {
            "type": "dense",
            "weights": np.random.randn(20, 5),
            "bias": np.random.randn(5),
        },
    ]
    return network


@pytest.fixture
def sample_dataset():
    """Fixture providing sample data for testing."""
    X = np.random.randn(100, 10)  # 100 samples, 10 features
    y = np.random.randint(0, 5, size=(100,))  # 5 classes
    return X, y
