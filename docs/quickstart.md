## Basic Usage


```python
from evolite import EvoLite

# Load your neural network model and dataset
model = ... # Your model here
data = ... # Your data here

# Create an EvoLite instance
evolite = EvoLite(model, data)

# Compress the model
compressed_model = evolite.compress()
```