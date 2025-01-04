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

## Quick Example

```python
from evolite import compress_network
from evolite.models import CompressibleNetwork

# Load your network
network = CompressibleNetwork()

# Compress it
compressed = compress_network(network, target_size=0.5)
```
