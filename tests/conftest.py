import pytest
import random
import numpy as np
import torch
import tensorflow as tf


@pytest.fixture(autouse=True)
def set_seed():
    """Set a fixed seed for all tests."""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tf.random.set_seed(seed)
