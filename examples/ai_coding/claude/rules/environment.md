# Environment Configuration

## VS Code Python Environment

**Always use the workspace's configured Python environment.**

This project uses the `FRAGNNET-GPU` conda environment configured in VS Code settings.

### Finding the Correct Python Interpreter

1. Check `.vscode/settings.json` for the configured interpreter path
2. Use the interpreter specified in VS Code settings, not system Python
3. If no setting exists, ask the user which environment to use

### Running Commands

**Always activate the correct environment before running Python commands:**

```bash
# Check which Python is configured
cat .vscode/settings.json | grep python

# Use the configured interpreter directly
/path/to/conda/envs/FRAGNNET-GPU/bin/python script.py

# Or activate conda environment first
conda activate FRAGNNET-GPU && python script.py
```

### Environment Verification

Before running tests or scripts, verify the environment:

```bash
# Check Python version (should be 3.10+)
python --version

# Verify key packages are installed
python -c "import torch; print(torch.__version__)"
python -c "import lightning; print(lightning.__version__)"
python -c "import dgl; print(dgl.__version__)"
```

### Do NOT

- Use system Python (`/usr/bin/python3`)
- Assume a virtual environment is activated
- Install packages globally
- Modify the conda environment without user confirmation

### HPC/SLURM Environments

For cluster jobs, use the environment scripts in `env/` directory:

```bash
# Load cluster-specific environment
source env/cluster_setup.sh
```

The SLURM scripts in `slurm_scripts/` already include proper environment activation.
