
# Import Conventions

## PyTorch Functional Module

**Always use uppercase `F` for the PyTorch functional module import:**

```python
import torch.nn.functional as F
```

**NOT:**
```python
# ❌ Incorrect - lowercase
import torch.nn.functional as f
```

### Usage

Use `F.` prefix for all functional operations:

```python
# Correct
normalized = F.normalize(tensor, p=2, dim=1)
normalized = F.softmax(tensor, dim=-1)

# Not this
normalized = f.normalize(tensor, p=2, dim=1)
```

### Rationale

- Follows PyTorch's official documentation and conventions
- Improves code readability and consistency with industry standards
- Matches ruff/pylint style preferences (PEP 8 naming conventions)
- Makes code more maintainable across the project

## Lightning Module

**Always use uppercase `L` for the Lightning import:**

```python
import lightning as L
```

**NOT:**
```python
# ❌ Incorrect - full module name
import lightning
import pytorch_lightning as pl
```

### Usage

Use `L.` prefix for all Lightning classes and utilities:

```python
# Correct
class MyModel(L.LightningModule):
    pass

trainer = L.Trainer(max_epochs=10)

# Not this
class MyModel(lightning.LightningModule):
    pass
```

### Rationale

- Follows Lightning's official naming convention (PyTorch Lightning rebranded as Lightning)
- Improves code readability with consistent aliases across the project
- Makes code more concise and easier to maintain
- Aligns with modern Lightning best practices
