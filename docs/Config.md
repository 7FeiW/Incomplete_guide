# Use Configurations
Configuration files are an effective way to track parameter setups and experiment configurations in your code base, ensuring reproducibility, clarity, and modularity. This practice allows you to separate code logic from experiment parameters, making it easier to modify, share, and understand experiments without directly altering source code.

## What could/should in the configureations file
In gernal, all mutable paremeters should stored in configuation files. Here is a list of item could/should keep in configureation files.

- **Paths to Input and Output Files:** Specify locations for datasets, model checkpoints, logs, and output results so that file management is consistent and easily adjustable between environments.
- **Experiment Metadata:** Information like experiment name, description, random seed for reproducibility, and version info for code or data.
- **Model Parameters:** Model architecture details, hyperparameters such as learning rate, batch size, number of epochs, dropout rates, optimizer choices, etc.
- **Data Parameters:** Details about data splits, preprocessing options, augmentation settings, or sample sizes that might vary between runs.
- **Training and Evaluation Settings:** Such as training schedules, early stopping criteria, metric names, evaluation intervals.
- **Environment and Hardware Settings:** GPU/CPU usage flags, number of workers/threads, and other execution environment details.
- **Debugging and Logging Options:** Verbosity level, log file paths, and whether to save intermediate results.
- **Random Seeds and Reproducibility Controls:** To make sure experiments can be repeated reliably.
- **External Service Credentials (if necessary):** Stored securely or referenced indirectly (such as API keys or database credentials), ideally outside version-controlled config files.
- **Configuration for Experiment Variants:** Ability to override or extend base configs for different experimental setups or environments (hierarchical config support).
- **Validation and Type Checking:** Include schema or validation rules so configs are verified before running experiments to avoid errors.
- **Documentation or Comments within Configs:** Useful especially for YAML, to clarify what each parameter controls.
- **Version and Dependency Information:** Details on code versions, dataset versions, or library dependencies to support experiment reproducibility.
- **Flags for Feature Toggles or Optional Modules:** To switch on/off certain parts of the code or models easily.

## Configureation File Format
Two popular text formats used for configuration files are json and yaml. 
### A Json Example
```json
{
  "experiment": {
    "name": "baseline_model_test",
    "seed": 42
  },
  "data": {
    "train_path": "data/train.csv",
    "val_path": "data/validate.csv",
    "batch_size": 64,
    "shuffle": true
  },
  "model": {
    "type": "resnet50",
    "dropout": 0.5
  },
  "training": {
    "epochs": 100,
    "learning_rate": 0.001,
    "optimizer": "adam"
  }
}

```
JSON is a widely used format, with built-in support in python. It is compact and uses explicit syntax with braces {}, brackets [], commas, and quotes. It does NOT support comments, which reduces readability for manual editing but makes it simpler and more predictable to parse programmatically.

### A Yaml example
```yaml
experiment:
  name: baseline_model_test
  seed: 42

data:
  train_path: data/train.csv
  val_path: data/validate.csv
  batch_size: 64
  shuffle: true

model:
  type: resnet50
  dropout: 0.5

training:
  epochs: 100
  learning_rate: 0.001
  optimizer: adam
```
YAML is designed primarily for human readability and uses indentation to structure data, which makes it easier to read and write by hand, especially for configuration files. It supports comments (using #), which is useful for documenting configurations directly.

## Load and Validate Configuratiuons

### Loading JSON and YAML config files
```python
import json
import yaml

def load_json_config(path):
    with open(path, 'r') as f:
        config = json.load(f)
    return config

def load_yaml_config(path):
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    return config


# Usage example:
json_config = load_json_config('config.json')
yaml_config = load_yaml_config('config.yaml')

print("JSON Config:", json_config)
print("YAML Config:", yaml_config)
```

### Validating JSON and YAML config files
TODO
```python
```

### Using a template configurtation for defualt values
Using a template configuration file for default values is a practice to standardize experiment setups, provide sensible defaults, and reduce duplication. This template can be a "base" config that your experiments start from, and you then override or extend it for specific runs or environments.

1. Create a Template Config File
Example YAML template, e.g., config-defaults.yaml:
```yaml
experiment:
  name: default_experiment
  seed: 12345

data:
  train_path: data/train.csv
  val_path: data/val.csv
  batch_size: 64
  shuffle: true

model:
  type: resnet50
  dropout: 0.5

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: adam

logging:
  verbosity: info
  save_checkpoints: true
  checkpoint_dir: checkpoints/
```
2. Create Experiment-Specific Override Config
Example config-exp-1.yaml:
```yaml
experiment:
  name: experiment1

training:
  epochs: 100
  learning_rate: 0.0005
```

3. Loading and Merging Template + Overrides in Python
You can use libraries like PyYAML and deepmerge or tools like OmegaConf or Hydra that support config composition and overrides.
```python
import yaml
from copy import deepcopy

def merge_dicts(base, override):
    """Recursively merge two dictionaries, with override taking precedence."""
    result = deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_dicts(result[k], v)
        else:
            result[k] = v
    return result

# Load defaults
with open('config-defaults.yaml', 'r') as f:
    defaults = yaml.safe_load(f)

# Load experiment-specific overrides
with open('config-experiment1.yaml', 'r') as f:
    overrides = yaml.safe_load(f)

# Merge configs
merged_config = merge_dicts(defaults, overrides)

print("Merged Config:")
print(merged_config)
```