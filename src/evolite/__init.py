"""EvoLite: Neural Network Compression through Evolutionary Computation."""

__version__ = "0.1.0"

from .core.compression import compress_network
from .core.evolution import EvolutionaryOptimizer
from .models.neural_net import CompressibleNetwork

__all__ = ["compress_network", "EvolutionaryOptimizer", "CompressibleNetwork"]
