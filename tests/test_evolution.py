import pytest
import numpy as np
from evolite.core.evolution import EvolutionaryOptimizer
from evolite.models.neural_net import CompressibleNetwork


def test_evolution_improves_fitness(sample_network, sample_dataset):
    """Test that evolution improves network fitness over generations."""
    X, y = sample_dataset
    optimizer = EvolutionaryOptimizer(population_size=20)

    def fitness_fn(network):
        return -np.mean((network.evaluate(X) - y) ** 2)  # Negative MSE

    initial_fitness = fitness_fn(sample_network)
    evolved = optimizer.evolve(sample_network, fitness_fn, n_generations=10)
    final_fitness = fitness_fn(evolved)

    assert final_fitness > initial_fitness


def test_population_diversity():
    """Test that evolution maintains population diversity."""
    optimizer = EvolutionaryOptimizer(population_size=20)
    optimizer.initialize_population(CompressibleNetwork())

    # Check that not all individuals are identical
    weights = [ind.layers[0]["weights"] for ind in optimizer.population[:5]]
    assert not all(np.array_equal(w1, w2) for w1 in weights for w2 in weights)


def test_evolution_parameters():
    """Test evolution with different parameter settings."""
    with pytest.raises(ValueError):
        EvolutionaryOptimizer(population_size=0)
    with pytest.raises(ValueError):
        EvolutionaryOptimizer(mutation_rate=-0.1)
    with pytest.raises(ValueError):
        EvolutionaryOptimizer(crossover_rate=1.5)
