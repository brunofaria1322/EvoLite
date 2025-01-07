# Welcome to EvoLite

Welcome to the **EvoLite** library! EvoLite is a Python library designed to minimize neural network sizes using model compression techniques combined with evolutionary computing. Whether you're optimizing for edge devices or simply want smaller, more efficient models, EvoLite has you covered.

## Features

- 🧬 Evolutionary-based network compression
- 🔧 Multiple compression techniques
- 📊 Performance monitoring and visualization
- 🔌 Easy integration with existing models

## Installation

```bash
pip install evolite
```

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