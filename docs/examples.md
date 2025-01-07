## Pruning

```python
from evolite import EvoLite
from evolite.compression import Pruning

# Load your neural network model and dataset
model = ... # Your model here
data = ... # Your data here

# Create an EvoLite instance
evolite = EvoLite(model, data)

# Compress the model using pruning
pruner = Pruning(rate=0.5)
compressed_model = pruner.apply(evolite)
```