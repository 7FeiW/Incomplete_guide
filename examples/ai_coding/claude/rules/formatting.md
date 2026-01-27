# Formatting and Linting

This project uses `ruff`, `black`, and `isort` for code formatting and linting.

## Line Length

**Maximum line length: 100 characters**

This applies to all tools (ruff, black, isort).

## Python Version

**Target: Python 3.10 - 3.13**

Use features available in Python 3.10+. The primary target for linting is Python 3.11.

## Running Formatters

```bash
# Check for linting issues
ruff check src/

# Auto-fix linting issues
ruff check src/ --fix

# Format code with black
black src/ --line-length 100

# Sort imports
isort src/
```

## Docstring Convention

**Use Google-style docstrings** for all public modules, classes, methods, and functions.

### Function Docstrings

```python
def compute_loss(pred: torch.Tensor, target: torch.Tensor, weight: float = 1.0) -> torch.Tensor:
    """Compute weighted cross-entropy loss.

    Args:
        pred: Predictions of shape (batch, num_classes).
        target: Ground truth labels of shape (batch,).
        weight: Loss weighting factor.

    Returns:
        Scalar loss tensor.

    Raises:
        ValueError: If pred and target batch dimensions don't match.
    """
```

### Class Docstrings

```python
class FragmentEncoder(nn.Module):
    """Encodes molecular fragments into fixed-size embeddings.

    This encoder uses a GNN to process fragment graphs and produces
    embeddings suitable for downstream prediction tasks.

    Attributes:
        hidden_dim: Dimension of hidden layers.
        output_dim: Dimension of output embeddings.
        num_layers: Number of GNN layers.

    Example:
        >>> encoder = FragmentEncoder(hidden_dim=128, output_dim=64)
        >>> fragment_graph = dgl.graph(...)
        >>> embeddings = encoder(fragment_graph)
    """

    def __init__(self, hidden_dim: int, output_dim: int, num_layers: int = 3):
        """Initialize the FragmentEncoder.

        Args:
            hidden_dim: Dimension of hidden layers.
            output_dim: Dimension of output embeddings.
            num_layers: Number of GNN layers.
        """
```

### Docstring Sections (in order)

Use these sections as needed, in this order:

1. **Summary line** - One-line description (required)
2. **Extended description** - Additional details (optional)
3. **Args** - Function/method arguments
4. **Returns** - Return value description
5. **Yields** - For generators
6. **Raises** - Exceptions that may be raised
7. **Attributes** - Class attributes (for class docstrings)
8. **Example** - Usage examples (optional but encouraged for complex APIs)
9. **Note** - Additional notes
10. **See Also** - Related functions/classes

### Tensor Shape Documentation

**Always document tensor shapes in Args and Returns:**

```python
def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    """Forward pass through the GNN layer.

    Args:
        x: Node features of shape (num_nodes, input_dim).
        edge_index: Edge connectivity of shape (2, num_edges).

    Returns:
        Updated node features of shape (num_nodes, output_dim).
    """
```

### Optional Arguments

Document default values and their meaning:

```python
def train_model(
    model: nn.Module,
    epochs: int = 100,
    lr: float = 1e-3,
    early_stopping: bool = True,
) -> dict[str, float]:
    """Train the model with optional early stopping.

    Args:
        model: The model to train.
        epochs: Maximum number of training epochs. Defaults to 100.
        lr: Learning rate for optimizer. Defaults to 1e-3.
        early_stopping: Whether to use early stopping based on
            validation loss. Defaults to True.

    Returns:
        Dictionary containing training metrics with keys:
        - "train_loss": Final training loss.
        - "val_loss": Final validation loss.
        - "epochs_trained": Actual number of epochs completed.
    """
```

### What NOT to Document

- Private methods (prefixed with `_`) - docstrings optional
- Obvious one-liner methods (e.g., simple getters)
- `self` parameter in methods
- `cls` parameter in class methods

## Import Ordering

Imports are sorted by `isort` with the `black` profile in this order:

1. `__future__` imports
2. Standard library (`os`, `sys`, `typing`, etc.)
3. Third-party packages (`torch`, `numpy`, `dgl`, etc.)
4. First-party (`fragnnet` package)
5. Local folder (relative imports)

**Example:**

```python
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from fragnnet.model.nn_blocks import MLP
from fragnnet.utils.data_utils import load_spectrum
```

## Ruff Lint Rules

The following rule sets are enabled:

| Code | Description |
|------|-------------|
| `E` | pycodestyle errors |
| `W` | pycodestyle warnings |
| `F` | pyflakes |
| `I` | isort |
| `B` | flake8-bugbear |
| `C4` | flake8-comprehensions |
| `UP` | pyupgrade |
| `ARG` | flake8-unused-arguments |
| `SIM` | flake8-simplify |
| `NPY` | NumPy-specific rules |

## Ignored Rules

These rules are intentionally disabled:

- `E501` - Line too long (handled by formatter)
- `B008` - Function calls in argument defaults (allowed for factory patterns)
- `ARG001/ARG002` - Unused arguments (common in callbacks/hooks)
- `SIM108` - Ternary operator (readability preference)

## Per-File Exceptions

- `__init__.py` - Unused imports (`F401`) allowed for re-exports
- `tests/**/*.py` - Relaxed rules for test files (asserts, unused args, magic values)
- `scripts/**/*.py` and `preproc_scripts/**/*.py` - More lenient for one-off scripts

## Unused Variables

Prefix unused variables with underscore to suppress warnings:

```python
# Good - underscore prefix indicates intentionally unused
for _ in range(10):
    pass

x, _unused, z = get_values()

# Bad - will trigger ARG/F841 warnings
for i in range(10):  # 'i' never used
    pass
```
