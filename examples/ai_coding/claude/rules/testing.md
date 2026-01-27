# Testing

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_models.py -v

# Run with coverage
pytest --cov=src tests/

# Run tests matching a pattern
pytest tests/ -v -k "test_forward"

# Run with verbose output on failure
pytest tests/ -v --tb=short
```

## Test Organization

- Place tests in `tests/` directory
- Name test files `test_<module>.py` (e.g., `test_fragnnet_model.py`)
- Group related tests by module, feature, or class
- One test file per major component

## Writing Tests

### Use pytest fixtures for setup

```python
@pytest.fixture
def sample_graph():
    """Create a sample DGL graph for testing."""
    return dgl.graph(([0, 1], [1, 2]))

def test_forward_pass(sample_graph):
    model = MyModel()
    output = model(sample_graph)
    assert output.shape == (3, 64)
```

### Test real values, not just existence

```python
# Good - tests specific expected behavior
def test_cosine_similarity():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    sim = cosine_similarity(a, b)
    assert sim == pytest.approx(0.0, abs=1e-6)

# Bad - only checks something exists
def test_cosine_similarity():
    result = cosine_similarity(a, b)
    assert result is not None  # Too weak!
```

### Include edge cases

```python
def test_empty_input():
    """Test behavior with empty tensor."""
    with pytest.raises(ValueError):
        model(torch.empty(0, 10))

def test_single_node_graph():
    """Test with minimal graph."""
    g = dgl.graph(([], []))
    g.ndata['feat'] = torch.randn(1, 32)
    output = model(g)
    assert output.shape[0] == 1
```

## What to Test

| Component | What to test |
|-----------|--------------|
| Models | Forward pass shapes, known input/output pairs |
| Loss functions | Known values, edge cases (zeros, large values) |
| Data loaders | Batch shapes, data types, edge cases |
| Utilities | Expected outputs for known inputs |
| Config parsing | Valid configs load, invalid configs raise errors |

## Assertions

- Use `pytest.approx()` for floating point comparisons
- Use `torch.allclose()` for tensor comparisons
- Check tensor shapes explicitly: `assert x.shape == (batch, dim)`
- Test device placement when relevant: `assert output.device == input.device`
