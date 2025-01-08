## API Reference

### `EvoLite`

- `__init__(model, dataset)`: Initialize EvoLite with a model and dataset.
- `compress()`: Compresses the model using EvoLite's evolutionary-based compression techniques.
- `save(path)`: Saves the compressed model to a file.

### Utilities

- `save_model(model, path)`: Save a PyTorch or TensorFlow model.
- `load_model(model, path)`: Load a PyTorch or TensorFlow model.
- `validate_dataset(dataset)`: Validate if the dataset is in the expected format.

### `Pruning`

- `__init__(target, rate)`: Initialize the pruning module.
- `apply(model)`: Apply pruning to the model.
